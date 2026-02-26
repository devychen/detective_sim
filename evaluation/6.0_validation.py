import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# 0. Settings
# =====================================================

mode = "aggregated"          # "aggregated" or "per_run"
include_optional = True      # include optional metrics

base_path = "evaluation/"

# =====================================================
# 1. Load data
# =====================================================

df_cls = pd.read_csv(base_path + "2.0_classifier_per_turn.csv")
df_lex = pd.read_csv(base_path + "3.1_lexical_similarity.csv")
df_dist = pd.read_csv(base_path + "3.2_intra_agent_distance.csv")
df_syn = pd.read_csv(base_path + "4.0_syntactic_results.csv")
df_sent = pd.read_csv(base_path + "5.2_sentiment_distance.csv")

# =====================================================
# 2. Aggregate if needed
# =====================================================

if mode == "aggregated":
    df_cls = df_cls.groupby(["case", "turn", "character"])["prob_correct"] \
                   .mean().reset_index(name="prob_mean")
    
    df_lex = df_lex.groupby(["case", "turn", "character"])["similarity"] \
                   .mean().reset_index()
    
    df_dist = df_dist.groupby(["case", "turn", "character"]) \
                     [["character_distance", "intra_similarity"]].mean().reset_index()
    
    df_syn = df_syn.groupby(["case", "turn", "character"]) \
                   [["depth", "z_ref", "z_sim"]].mean().reset_index()
    
    df_sent = df_sent.groupby(["case", "turn", "character"])["distance"] \
                     .mean().reset_index()
    
    merge_keys = ["case", "turn", "character"]

else:
    df_cls = df_cls.rename(columns={"prob_correct": "prob_mean"})
    merge_keys = ["case", "run_id", "turn", "character"]

# =====================================================
# 3. Merge all metrics (TURN-LEVEL DATASET)
# =====================================================

df = df_cls.merge(df_lex, on=merge_keys, how="left")
df = df.merge(df_dist, on=merge_keys, how="left")
df = df.merge(df_syn, on=merge_keys, how="left")
df = df.merge(df_sent, on=merge_keys, how="left")

# 保存 turn-level 对齐数据
df.to_csv(
    f"evaluation/6.0_validation_turn_level_{mode}.csv",
    index=False,
    float_format="%.3f"
)

# =====================================================
# 4. Validation statistics
# =====================================================

def compute_correlations(x, y):
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue

def compute_ece(confidence, target, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidence, bins) - 1
    
    ece = 0.0
    N = len(confidence)
    
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            conf_bin = np.mean(confidence[mask])
            acc_bin = np.mean(target[mask])
            ece += (np.sum(mask) / N) * abs(conf_bin - acc_bin)
    
    return ece

mandatory_metrics = {
    "lexical_similarity": "similarity",
    "character_distance": "character_distance",
    "z_ref": "z_ref",
    "sentiment_distance": "distance"
}

optional_metrics = {
    "intra_similarity": "intra_similarity",
    "depth": "depth",
    "z_sim": "z_sim"
}

metrics = mandatory_metrics.copy()
if include_optional:
    metrics.update(optional_metrics)

results = []

for name, col in metrics.items():
    
    if col not in df.columns:
        continue
    
    sub = df[[col, "prob_mean"]].dropna()
    
    if len(sub) < 5:
        continue
    
    r, rp, rho, rhop = compute_correlations(sub[col], sub["prob_mean"])
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sub[[col]]).flatten()
    
    ece = compute_ece(scaled, sub["prob_mean"])
    
    results.append({
        "metric": name,
        "n_samples": len(sub),
        "pearson_r": r,
        "pearson_p": rp,
        "spearman_rho": rho,
        "spearman_p": rhop,
        "ECE": ece
    })

summary_df = pd.DataFrame(results)

# 保存 summary
summary_df.to_csv(
    f"evaluation/6.0_validation_summary_{mode}.csv",
    index=False,
    float_format="%.3f"
)

# 打印 summary
print(f"\nMode: {mode} | Include optional: {include_optional}")
print(summary_df.to_string(index=False, float_format="{:.3f}".format))

# =====================================================
# 5. Faithfulness ranking
# =====================================================

ranking_df = summary_df.copy()

# 取绝对相关
ranking_df["abs_pearson"] = ranking_df["pearson_r"].abs()
ranking_df["abs_spearman"] = ranking_df["spearman_rho"].abs()

# 归一化到 0-1
for col in ["abs_pearson", "abs_spearman"]:
    min_val = ranking_df[col].min()
    max_val = ranking_df[col].max()
    if max_val > min_val:
        ranking_df[col] = (ranking_df[col] - min_val) / (max_val - min_val)
    else:
        ranking_df[col] = 0

# ECE 越小越好 → 转成 calibration score
ranking_df["calibration_score"] = 1 - ranking_df["ECE"]

min_val = ranking_df["calibration_score"].min()
max_val = ranking_df["calibration_score"].max()
if max_val > min_val:
    ranking_df["calibration_score"] = (
        (ranking_df["calibration_score"] - min_val) /
        (max_val - min_val)
    )
else:
    ranking_df["calibration_score"] = 0

# Composite Faithfulness Score
ranking_df["faithfulness_score"] = (
    ranking_df["abs_pearson"] +
    ranking_df["abs_spearman"] +
    ranking_df["calibration_score"]
) / 3

ranking_df = ranking_df.sort_values(
    by="faithfulness_score",
    ascending=False
)

# 只保留核心列
ranking_output = ranking_df[[
    "metric",
    "faithfulness_score"
]]

# 保存
ranking_output.to_csv(
    f"evaluation/6.0_validation_ranking_{mode}.csv",
    index=False,
    float_format="%.3f"
)

print("\n=== Faithfulness Ranking ===")
print(ranking_output.to_string(index=False, float_format="{:.3f}".format))