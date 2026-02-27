"""
=========================================================
2.0_classifier.py 

PIPELINE OVERVIEW
-----------------
This script evaluates whether LLM agents remain in-character (IC) 
by using a trained BERT-based classifier as a baseline.

For each dialogue simulation, the classifier predicts the probability 
that an utterance belongs to the correct character (Holmes, Poirot, Marple).

We compute two metrics:
1. prob_correct: predicted probability of the true character class.
2. brier score: squared error of the probability prediction.

The evaluation consists of three main stages:

---------------------------------------------------------
(1) Turn-level descriptive statistics with Bootstrap CIs
---------------------------------------------------------
- Align turns across simulation runs.
- Compute mean prob_correct and brier score per turn.
- Estimate 95% confidence intervals using non-parametric bootstrap.
- Purpose: visualize temporal trends and uncertainty.

Methods / Libraries:
- numpy, pandas
- bootstrap resampling

---------------------------------------------------------
(2) Pooled regression analysis (turn-level trend inference)
---------------------------------------------------------
- Fit a single linear regression model across all runs:
      metric ~ turn
- Each utterance is treated as one observation.
- Use Ordinary Least Squares (OLS) from statsmodels.
- Extract slope, t-statistic, p-value, confidence interval, R².
- Purpose: test whether character consistency changes over turns.

Methods / Libraries:
- statsmodels.api.OLS

---------------------------------------------------------
(3) Outputs
---------------------------------------------------------
- CSV files:
    * turn-level aggregated statistics with bootstrap CIs
    * pooled regression results
- Plots:
    * turn vs metric curves with bootstrap confidence bands

---------------------------------------------------------
Packages / Libraries Used
---------------------------------------------------------
- transformers (HuggingFace): BERT classifier
- torch: GPU inference
- pandas, numpy: data processing
- statsmodels: statistical regression
- matplotlib, seaborn: visualization

=========================================================
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

import statsmodels.api as sm  # For pooled OLS regression

from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
        raise FileNotFoundError("No checkpoint found.")
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
    """
    Run BERT classifier and return class probabilities.
    """
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
    For one simulation run, compute metrics per agent and per turn.

    Output columns:
        turn
        prob_correct
        brier
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
# Aggregate simulation runs
# =================================================

def aggregate_over_runs(all_runs: List[Dict[str, pd.DataFrame]]):
    """
    Collect per-run DataFrames for each agent.
    """
    aggregated = {agent: [] for agent in AGENTS}
    for run in all_runs:
        for agent, df in run.items():
            aggregated[agent].append(df)
    return aggregated


# =================================================
# Bootstrap CI helper
# =================================================

def bootstrap_ci(values, n_boot=10000, ci=95):
    """
    Compute bootstrap confidence interval for the mean.
    """

    values = np.array(values)
    means = []

    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())

    means = np.array(means)

    alpha = (100 - ci) / 2
    ci_low = np.percentile(means, alpha)
    ci_high = np.percentile(means, 100 - alpha)

    return values.mean(), ci_low, ci_high


# =================================================
# Turn-level aggregation with Bootstrap CIs
# =================================================

def aggregate_turn_level(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Align turns across simulation runs and compute mean + bootstrap CI.
    """

    all_data = pd.concat(dfs, ignore_index=True)
    results = []

    for turn, group in all_data.groupby("turn"):
        prob_mean, prob_low, prob_high = bootstrap_ci(group["prob_correct"])
        brier_mean, brier_low, brier_high = bootstrap_ci(group["brier"])

        results.append({
            "turn": turn,
            "prob_mean": prob_mean,
            "prob_ci_low": prob_low,
            "prob_ci_high": prob_high,
            "brier_mean": brier_mean,
            "brier_ci_low": brier_low,
            "brier_ci_high": brier_high,
            "n_obs": len(group)
        })

    return pd.DataFrame(results).sort_values("turn")


# =================================================
# Pooled regression (statsmodels OLS)
# =================================================

def pooled_regression(dfs: List[pd.DataFrame], metric: str):
    """
    Perform pooled linear regression:
        metric ~ turn

    All utterances across all runs are treated as observations.
    """

    all_data = pd.concat(dfs, ignore_index=True)

    X = sm.add_constant(all_data["turn"])  # add intercept
    y = all_data[metric]

    model = sm.OLS(y, X).fit()

    slope = model.params["turn"]
    p_value = model.pvalues["turn"]
    t_value = model.tvalues["turn"]
    ci_low, ci_high = model.conf_int().loc["turn"]

    return {
        "slope": slope,
        "p_value": p_value,
        "t_value": t_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": len(all_data),
        "r2": model.rsquared
    }


# =================================================
# Plotting
# =================================================

def plot_with_ci(df, metric_prefix, case_name, agent, output_dir):
    plt.figure(figsize=(8, 5))

    mean = df[f"{metric_prefix}_mean"]
    low  = df[f"{metric_prefix}_ci_low"]
    high = df[f"{metric_prefix}_ci_high"]

    plt.plot(df["turn"], mean, marker="o", label=f"{metric_prefix} mean")
    plt.fill_between(df["turn"], low, high, alpha=0.3, label="Bootstrap CI")

    plt.xlabel("Turn")
    plt.ylabel(metric_prefix)
    plt.title(f"{case_name} - {agent} - {metric_prefix} over turns")
    plt.legend()

    save_path = os.path.join(output_dir, f"{case_name}_{agent}_{metric_prefix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =================================================
# Case-level evaluation
# =================================================
def evaluate_case(case_name, tokenizer, model, f,
                  prob_reg_results, brier_reg_results,
                  global_per_turn_records):

    pattern = os.path.join(DATA_ROOT, case_name, "run_*", "dialogue_log.csv")
    dialogue_logs = sorted(glob.glob(pattern))

    print(f"\n=== {case_name.upper()} ===", file=f)
    print(f"Found {len(dialogue_logs)} simulation runs.", file=f)

    all_runs = []

    for path in dialogue_logs:
        run_id = os.path.basename(os.path.dirname(path))
        run_results = evaluate_simulation(path, tokenizer, model)
        all_runs.append(run_results)

        # Collect per-turn raw data (for CSV)
        for agent, df_agent in run_results.items():
            for _, row in df_agent.iterrows():
                global_per_turn_records.append({
                    "case": case_name,
                    "run_id": run_id,
                    "turn": int(row["turn"]),
                    "character": agent.lower(),
                    "prob_correct": row["prob_correct"],
                    "brier": row["brier"]
                })

    aggregated = aggregate_over_runs(all_runs)

    turn_csv_dir = "./evaluation/turn_csv_classifier"
    reg_csv_dir = "./evaluation/regression_csv_classifier"
    plots_dir = "./evaluation/plots_classifier"

    os.makedirs(turn_csv_dir, exist_ok=True)
    os.makedirs(reg_csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    agent_turn_stats = {}

    # ------------------------------
    # Agent-level stats + regression
    # ------------------------------
    for agent, dfs in aggregated.items():
        if len(dfs) == 0:
            continue

        print(f"\n--- {agent} ---", file=f)

        turn_df = aggregate_turn_level(dfs)
        agent_turn_stats[agent] = turn_df

        print("\nTurn-level aggregated metrics (Bootstrap 95% CI):", file=f)
        print(turn_df.to_string(index=False, float_format="%.4f"), file=f)

        turn_df.to_csv(
            os.path.join(turn_csv_dir, f"{case_name}_{agent}_turn_stats.csv"),
            index=False
        )

        prob_reg = pooled_regression(dfs, "prob_correct")
        brier_reg = pooled_regression(dfs, "brier")

        pd.DataFrame([
            {"metric": "prob_correct", **prob_reg},
            {"metric": "brier", **brier_reg}
        ]).to_csv(
            os.path.join(reg_csv_dir, f"{case_name}_{agent}_pooled_regression.csv"),
            index=False
        )

        prob_reg_results.append({
            "Case": case_name,
            "Character": agent,
            "Beta": prob_reg["slope"],
            "CI_low": prob_reg["ci_low"],
            "CI_high": prob_reg["ci_high"],
            "p_value": prob_reg["p_value"],
            "R2": prob_reg["r2"]
        })

        brier_reg_results.append({
            "Case": case_name,
            "Character": agent,
            "Beta": brier_reg["slope"],
            "CI_low": brier_reg["ci_low"],
            "CI_high": brier_reg["ci_high"],
            "p_value": brier_reg["p_value"],
            "R2": brier_reg["r2"]
        })

    # ------------------------------
    # Multi-agent plots (6 total)
    # ------------------------------
    for metric in ["prob", "brier"]:

        plt.figure(figsize=(8, 6))

        for agent in AGENTS:

            if agent not in agent_turn_stats:
                continue

            df_plot = agent_turn_stats[agent]
            x = df_plot["turn"]

            if metric == "prob":
                y = df_plot["prob_mean"]
                ci_low = df_plot["prob_ci_low"]
                ci_high = df_plot["prob_ci_high"]
                ylabel = "Probability (Correct Character)"
            else:
                y = df_plot["brier_mean"]
                ci_low = df_plot["brier_ci_low"]
                ci_high = df_plot["brier_ci_high"]
                ylabel = "Brier Score"

            plt.plot(x, y,
                     label=agent,
                     color=AGENT_COLORS[agent],
                     linewidth=2)

            plt.fill_between(
                x,
                ci_low,
                ci_high,
                color=AGENT_COLORS[agent],
                alpha=0.2
            )

        plt.xlabel("Turn")
        plt.ylabel(ylabel)
        plt.title(f"{case_name.upper()} – {ylabel} over Turns")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(
            plots_dir,
            f"{case_name}_{metric}_multi_agent.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved plot: {plot_path}")

# =================================================
# Main
# =================================================

def main():
    np.random.seed(42)
    tokenizer, model = load_classifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./evaluation"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"2.0_classifier_{timestamp}.txt")

    prob_reg_results = []
    brier_reg_results = []
    global_per_turn_records = []

    with open(output_path, "w", encoding="utf-8") as f:

        for case in CASES:
            evaluate_case(case, tokenizer, model, f,
                          prob_reg_results,
                          brier_reg_results,
                          global_per_turn_records)

        # ===============================
        # Write per-turn CSV (OVERWRITE)
        # ===============================
        per_turn_df = pd.DataFrame(global_per_turn_records)
        per_turn_output_path = os.path.join(
            output_dir,
            "2.0_classifier_per_turn.csv"
        )
        per_turn_df.to_csv(per_turn_output_path, index=False)

        print(f"\nPer-turn CSV saved to {per_turn_output_path}", file=f)

        # ===============================
        # Final regression tables
        # ===============================

        print("\n\n==============================", file=f)
        print("Regression results for prob_correct", file=f)
        print("==============================\n", file=f)

        for row in prob_reg_results:
            ci_str = f"[{row['CI_low']:.3f}, {row['CI_high']:.3f}]"
            p_str = format_p_value(row["p_value"])
            print(f"{row['Case']}\t{row['Character']}\t"
                  f"{row['Beta']:.3f}\t{ci_str}\t"
                  f"{p_str}\t{row['R2']:.3f}",
                  file=f)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
