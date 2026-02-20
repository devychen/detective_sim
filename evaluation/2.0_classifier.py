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

import statsmodels.api as sm
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

AGENT_COLORS = {
    "Holmes": "#1f77b4",
    "Marple": "#ff7f0e",
    "Poirot": "#2ca02c"
}


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
    aggregated = {agent: [] for agent in AGENTS}
    for run in all_runs:
        for agent, df in run.items():
            aggregated[agent].append(df)
    return aggregated


# =================================================
# Bootstrap CI helper
# =================================================

def bootstrap_ci(values, n_boot=10000, ci=95):
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
# Turn-level aggregation
# =================================================

def aggregate_turn_level(dfs: List[pd.DataFrame]) -> pd.DataFrame:
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
# Pooled regression
# =================================================

def pooled_regression(dfs: List[pd.DataFrame], metric: str):
    all_data = pd.concat(dfs, ignore_index=True)

    X = sm.add_constant(all_data["turn"])
    y = all_data[metric]

    model = sm.OLS(y, X).fit()

    slope = model.params["turn"]
    p_value = model.pvalues["turn"]
    ci_low, ci_high = model.conf_int().loc["turn"]

    return {
        "slope": slope,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": len(all_data),
        "r2": model.rsquared
    }


# =================================================
# Format p-value with stars
# =================================================

def format_p_value(p):
    if p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


# =================================================
# Case-level evaluation
# =================================================

def evaluate_case(case_name, tokenizer, model, f,
                  prob_reg_results, brier_reg_results):

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

        print("\nTurn-level aggregated metrics (Bootstrap 95% CI):", file=f)
        print(turn_df.to_string(index=False, float_format="%.4f"), file=f)

        turn_df.to_csv(
            os.path.join(turn_csv_dir, f"{case_name}_{agent}_turn_stats.csv"),
            index=False
        )

        prob_reg = pooled_regression(dfs, "prob_correct")
        brier_reg = pooled_regression(dfs, "brier")

        # Save regression CSV (unchanged)
        pd.DataFrame([
            {"metric": "prob_correct", **prob_reg},
            {"metric": "brier", **brier_reg}
        ]).to_csv(
            os.path.join(reg_csv_dir, f"{case_name}_{agent}_pooled_regression.csv"),
            index=False
        )

        # Store for final integrated table
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

    with open(output_path, "w", encoding="utf-8") as f:

        for case in CASES:
            evaluate_case(case, tokenizer, model, f,
                          prob_reg_results, brier_reg_results)

        # ==============================
        # Final integrated regression tables
        # ==============================

        print("\n\n==============================", file=f)
        print("Regression results for prob_correct", file=f)
        print("==============================\n", file=f)

        print("Case\tCharacter\tβ (prob_correct)\t95% CI\tp-value\tR²", file=f)

        for row in prob_reg_results:
            ci_str = f"[{row['CI_low']:.3f}, {row['CI_high']:.3f}]"
            p_str = format_p_value(row["p_value"])

            print(f"{row['Case']}\t"
                  f"{row['Character']}\t"
                  f"{row['Beta']:.3f}\t"
                  f"{ci_str}\t"
                  f"{p_str}\t"
                  f"{row['R2']:.3f}",
                  file=f)

        print("\n\n==============================", file=f)
        print("Regression results for Brier score", file=f)
        print("==============================\n", file=f)

        print("Case\tCharacter\tβ (brier)\t95% CI\tp-value\tR²", file=f)

        for row in brier_reg_results:
            ci_str = f"[{row['CI_low']:.3f}, {row['CI_high']:.3f}]"
            p_str = format_p_value(row["p_value"])

            print(f"{row['Case']}\t"
                  f"{row['Character']}\t"
                  f"{row['Beta']:.3f}\t"
                  f"{ci_str}\t"
                  f"{p_str}\t"
                  f"{row['R2']:.3f}",
                  file=f)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
