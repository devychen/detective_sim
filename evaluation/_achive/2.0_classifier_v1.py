"""
2.0_classifier.py (Revised Version)

PER-RUN REGRESSION

Implements advisor-recommended improvements:

1. Turn-level confidence intervals now computed using Bootstrap (10,000 samples).
   This avoids invalid CI bounds (below 0 or above 1) and makes no normality assumption.

2. Regression analysis now performed per-run (per simulation):
   - Each run provides an independent slope estimate (turn → metric)
   - A one-sample t-test is performed over all slopes
   This ensures statistical validity (independent samples).
   X-turn numbers, Y-accuracy

Bootstrap CI, Per-run slope + one-sample t-test

Output:
- Compute slopes per-run
- Do one-sample t-test over slopes
- Compute aggregated prob means per turn
- Plot aggregated prob mean + CI
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
from scipy.stats import ttest_1samp

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

    Produces:
        turn
        prob_correct (model predicted prob of correct character)
        brier (squared error for probability)
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
    Collects a list of (per-run DataFrames) for each agent.
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
    Computes bootstrap confidence interval for an array of values.

    values: array-like of numbers
    n_boot: number of bootstrap samples
    ci: bootstrap confidence level

    Returns:
        mean, ci_low, ci_high
    """

    values = np.array(values)
    means = []

    # Bootstrap resampling
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
    Align turns across simulation runs and compute:
      - mean prob_correct + bootstrap CI
      - mean brier + bootstrap CI

    Bootstrap is used instead of normal-based CI to ensure bounded intervals
    and avoid assumptions about the distribution of probabilities.
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
            "n_runs": len(group)
        })

    df_out = pd.DataFrame(results).sort_values("turn")
    return df_out


# =================================================
# Per-run regression slopes + one-sample t-test
# =================================================

def regression_per_run(dfs: List[pd.DataFrame], metric: str):
    """
    Performs regression per simulation run, then tests whether the mean slope differs from 0.

    Steps:
      1. For each run's DataFrame:
           regress: metric ~ turn
      2. Collect all slopes
      3. One-sample t-test: H0: mean slope = 0

    Returns:
      slope_mean, slope_std, n_runs, p_value
    """

    slopes = []

    for df in dfs:
        # Need at least 2 points to fit regression
        if df["turn"].nunique() < 2:
            continue

        X = df["turn"].values.reshape(-1, 1)
        y = df[metric].values

        reg = LinearRegression().fit(X, y)
        slopes.append(reg.coef_[0])

    slopes = np.array(slopes)
    n = len(slopes)

    if n < 2:
        return np.nan, np.nan, n, np.nan

    t_stat, p_value = ttest_1samp(slopes, popmean=0)

    return slopes.mean(), slopes.std(ddof=1), n, p_value


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

def evaluate_case(case_name, tokenizer, model, f):

    pattern = os.path.join(DATA_ROOT, case_name, "run_*", "dialogue_log.csv")
    dialogue_logs = sorted(glob.glob(pattern))

    print(f"\n=== {case_name.upper()} ===", file=f)
    print(f"Found {len(dialogue_logs)} simulation runs.", file=f)

    # Process each simulation run
    all_runs = []
    for path in dialogue_logs:
        all_runs.append(evaluate_simulation(path, tokenizer, model))

    # Aggregate runs per agent
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

        # Turn-level statistics with bootstrap CIs
        turn_df = aggregate_turn_level(dfs)

        # Print table
        print("\nTurn-level aggregated metrics (Bootstrap 95% CI):", file=f)
        print(
            turn_df.to_string(index=False, float_format="%.4f"),
            file=f
        )

        # Save CSV
        turn_df.to_csv(
            os.path.join(turn_csv_dir, f"{case_name}_{agent}_turn_stats.csv"),
            index=False
        )

        # Per-run regression (slope distribution)
        prob_slope_mean, prob_slope_std, n_prob, prob_p = regression_per_run(dfs, "prob_correct")
        brier_slope_mean, brier_slope_std, n_brier, brier_p = regression_per_run(dfs, "brier")

        print("\nPer-run regression statistics:", file=f)
        print(f"Probability slope mean: {prob_slope_mean:.4f}  (std={prob_slope_std:.4f}, n={n_prob})", file=f)
        print(f"Probability slope p-value: {prob_p:.4g}", file=f)
        print(f"Brier slope mean: {brier_slope_mean:.4f}  (std={brier_slope_std:.4f}, n={n_brier})", file=f)
        print(f"Brier slope p-value: {brier_p:.4g}", file=f)

        # Save regression summary CSV
        pd.DataFrame({
            "metric": ["prob", "brier"],
            "slope_mean": [prob_slope_mean, brier_slope_mean],
            "slope_std": [prob_slope_std, brier_slope_std],
            "n_runs": [n_prob, n_brier],
            "p_value": [prob_p, brier_p]
        }).to_csv(
            os.path.join(reg_csv_dir, f"{case_name}_{agent}_regression.csv"),
            index=False
        )

        # Plots
        plot_with_ci(turn_df, "prob", case_name, agent, plots_dir)
        plot_with_ci(turn_df, "brier", case_name, agent, plots_dir)


# =================================================
# Main
# =================================================

def main():
    tokenizer, model = load_classifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./evaluation"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"2.0_classifier_{timestamp}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for case in CASES:
            evaluate_case(case, tokenizer, model, f)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
