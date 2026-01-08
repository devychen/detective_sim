"""
evaluate_classifier.py

Purely analysis, no model training.

Evaluate character consistency (OOC) using a trained baseline classifier.

Implements advisor-suggested analyses:

1. Turn-level classification confidence (mean + CI over simulations)
2. Turn-level Brier score (mean + CI over simulations)
3. Linear regression on aggregated trajectories to detect drift

Extended:
- Automatically evaluates multiple cases (case1, case2, case3)
- Outputs detailed turn-level tables + regression summaries
"""

# =================================================
# Imports
# =================================================

import os
import glob
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


from datetime import datetime
from typing import List, Dict
from scipy.stats import t

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.linear_model import LinearRegression

# =================================================
# Configuration
# =================================================

MODEL_DIR = "./models/3class"

CASES = ["case1", "case2", "case3"]
DATA_ROOT = "./data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ID2LABEL = {
    0: "holmes",
    1: "marple",
    2: "poirot"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

AGENTS = ["Holmes", "Marple", "Poirot"]

# =================================================
# Helper: load latest checkpoint
# =================================================

def load_latest_checkpoint(model_dir: str):
    checkpoints = sorted(
        glob.glob(os.path.join(model_dir, "checkpoint-*")),
        key=os.path.getmtime
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in model directory.")
    return checkpoints[-1]

# =================================================
# Load classifier
# =================================================

def load_classifier():
    checkpoint = load_latest_checkpoint(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# =================================================
# Prediction
# =================================================

def predict_probabilities(texts, tokenizer, model):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)

    return probs.cpu().numpy()

# =================================================
# Per-simulation evaluation
# =================================================

def evaluate_simulation(dialogue_path, tokenizer, model) -> Dict[str, pd.DataFrame]:
    """
    Returns per-agent turn-level metrics for one simulation.
    """
    df = pd.read_csv(dialogue_path)
    results = {}

    for agent in AGENTS:
        agent_df = df[df["speaker"] == agent].copy()
        if agent_df.empty:
            continue

        texts = agent_df["utterance"].tolist()
        turns = agent_df["turn"].tolist()

        probs = predict_probabilities(texts, tokenizer, model)
        correct_id = LABEL2ID[agent.lower()]
        prob_correct = probs[:, correct_id]
        brier = (prob_correct - 1.0) ** 2

        results[agent] = pd.DataFrame({
            "turn": turns,
            "prob_correct": prob_correct,
            "brier": brier
        })

    return results

# =================================================
# Aggregate simulations (agent-wise)
# =================================================

def aggregate_over_runs(all_runs: List[Dict[str, pd.DataFrame]]):
    aggregated = {agent: [] for agent in AGENTS}
    for run in all_runs:
        for agent, df in run.items():
            aggregated[agent].append(df)
    return aggregated
    

# =================================================
# Turn-level aggregation (mean + CI)
# =================================================

def aggregate_turn_level(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Align turns across simulations and compute mean + 95% CI
    for prob_correct and brier score.
    """
    all_data = pd.concat(dfs, ignore_index=True)

    counts = (
        all_data
        .groupby("turn")
        .size()
        .rename("n_runs")
    )

    def mean_ci(x):
        mean = x.mean()
        sem = x.std(ddof=1) / np.sqrt(len(x))
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
        return pd.Series({
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

    prob_stats = (
        all_data
        .groupby("turn")["prob_correct"]
        .apply(mean_ci)
        .unstack()
        .add_prefix("prob_")
    )

    brier_stats = (
        all_data
        .groupby("turn")["brier"]
        .apply(mean_ci)
        .unstack()
        .add_prefix("brier_")
    )

    aggregated = (
        pd.concat([prob_stats, brier_stats], axis=1)
        .join(counts)          # 把 n_runs 加进来
        .reset_index()
        .sort_values("turn")
    )

    return aggregated

# =================================================
# Regression on aggregated trajectories
# =================================================

def regression_on_aggregated(df: pd.DataFrame, metric_prefix: str):
    X = df["turn"].values.reshape(-1, 1)
    y = df[f"{metric_prefix}_mean"].values

    if len(np.unique(X)) < 2:
        return np.nan, np.nan

    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]

    y_pred = reg.predict(X)
    residuals = y - y_pred
    n = len(y)

    if n < 3:
        return slope, np.nan

    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_var = np.sum((X - X.mean()) ** 2)
    se_slope = s_err / np.sqrt(x_var)

    t_stat = slope / se_slope
    dfree = n - 2

    # two-sided p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=dfree))

    return slope, p_value

# =================================================
# Case-level evaluation
# =================================================

def evaluate_case(case_name, tokenizer, model, f):
    pattern = os.path.join(DATA_ROOT, case_name, "run_*", "dialogue_log.csv")
    dialogue_logs = sorted(glob.glob(pattern))

    print(f"\n=== {case_name.upper()} ===", file=f)
    print(f"Found {len(dialogue_logs)} simulation runs.", file=f)

    all_runs = []
    for path in dialogue_logs:
        all_runs.append(evaluate_simulation(path, tokenizer, model))

    aggregated = aggregate_over_runs(all_runs)

    turn_csv_dir = "./evaluation/turn_csv"
    reg_csv_dir = "./evaluation/regression_csv"
    plots_dir = "./evaluation/plots"
    os.makedirs(turn_csv_dir, exist_ok=True)
    os.makedirs(reg_csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for agent, dfs in aggregated.items():
        if len(dfs) == 0:
            continue

        print(f"\n--- {agent} ---", file=f)

        turn_df = aggregate_turn_level(dfs)

        # only turns with >= 3 runs for regression
        turn_df_reg = turn_df[turn_df["n_runs"] >= 3]

        # print turn table
        print("\nTurn-level aggregated metrics (mean ± 95% CI):", file=f)
        print(
            turn_df.to_string(
                index=False,
                float_format="%.4f"
            ),
            file=f
        )

        # save turn-level CSV
        turn_df.to_csv(
            os.path.join(turn_csv_dir, f"{case_name}_{agent}_turn_stats.csv"),
            index=False
        )

        # regression analysis
        prob_slope, prob_p = regression_on_aggregated(turn_df_reg, "prob")
        brier_slope, brier_p = regression_on_aggregated(turn_df_reg, "brier")

        # print regression
        print("\nRegression on aggregated trajectories:", file=f)
        print(f"Probability slope: {prob_slope:.4f}", file=f)
        print(f"Probability p-value: {prob_p:.4g}", file=f)
        print(f"Brier slope: {brier_slope:.4f}", file=f)
        print(f"Brier p-value: {brier_p:.4g}", file=f)

        # save regression CSV
        reg_summary = pd.DataFrame({
            "metric": ["prob", "brier"],
            "slope": [prob_slope, brier_slope],
            "p_value": [prob_p, brier_p]
        })
        reg_summary.to_csv(
            os.path.join(reg_csv_dir, f"{case_name}_{agent}_regression.csv"),
            index=False
        )

        # plots
        plot_with_ci(turn_df, "prob", case_name, agent, plots_dir)
        plot_with_ci(turn_df, "brier", case_name, agent, plots_dir)

# =================================================
# Picture drawing
# =================================================


def plot_with_ci(df, metric_prefix, case_name, agent, output_dir):
    """
    Draws mean ± CI trend plots for prob_correct and brier over turns.
    """
    plt.figure(figsize=(8, 5))

    mean = df[f"{metric_prefix}_mean"]
    low  = df[f"{metric_prefix}_ci_low"]
    high = df[f"{metric_prefix}_ci_high"]

    plt.plot(df["turn"], mean, marker="o", label=f"{metric_prefix} mean")
    plt.fill_between(df["turn"], low, high, alpha=0.3, label="95% CI")

    plt.xlabel("Turn")
    plt.ylabel(metric_prefix)
    plt.title(f"{case_name} - {agent} - {metric_prefix} over turns")
    plt.legend()

    save_path = os.path.join(output_dir, f"{case_name}_{agent}_{metric_prefix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



# =================================================
# Entry point
# =================================================

def main():
    tokenizer, model = load_classifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "./evaluation"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"results_model_{timestamp}.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for case_name in CASES:
            evaluate_case(case_name, tokenizer, model, f)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
