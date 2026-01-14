"""
evaluate_model.py

Purely analysis, no model training.

Evaluate character consistency (OOC) using a trained baseline classifier.

This script implements the three evaluation measures suggested by the advisor:

1. Classification confidence over dialogue turns
2. Brier score over dialogue turns
3. Linear regression analysis to detect systematic drift over time

Extended version:
- Automatically evaluates multiple cases (case1, case2, case3) in one run.
"""

# =================================================
# Imports
# =================================================

import os
import glob
import numpy as np
import pandas as pd

from datetime import datetime

from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp

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


def compute_brier_score(prob_correct: float) -> float:
    return (prob_correct - 1.0) ** 2


# =================================================
# Per-run evaluation
# =================================================

def evaluate_simulation(dialogue_path, tokenizer, model):
    df = pd.read_csv(dialogue_path)
    results = {}

    for agent in ["Holmes", "Marple", "Poirot"]:
        agent_df = df[df["speaker"] == agent].copy()
        if agent_df.empty:
            continue

        texts = agent_df["utterance"].tolist()
        turns = agent_df["turn"].tolist()

        probs = predict_probabilities(texts, tokenizer, model)

        correct_id = LABEL2ID[agent.lower()]
        prob_correct = probs[:, correct_id]
        brier = [(p - 1.0) ** 2 for p in prob_correct]

        results[agent] = pd.DataFrame({
            "turn": turns,
            "prob_correct": prob_correct,
            "brier": brier
        })

    return results


def aggregate_over_runs(all_runs):
    aggregated = {a: [] for a in ["Holmes", "Marple", "Poirot"]}
    for run in all_runs:
        for agent, df in run.items():
            aggregated[agent].append(df)
    return aggregated


# =================================================
# Regression
# =================================================

def regression_slope_test(dfs, metric):
    slopes = []

    for df in dfs:
        X = df["turn"].values.reshape(-1, 1)
        y = df[metric].values
        if len(np.unique(X)) < 2:
            continue
        reg = LinearRegression().fit(X, y)
        slopes.append(reg.coef_[0])

    slopes = np.array(slopes)
    if len(slopes) == 0:
        return slopes, np.nan

    _, p_value = ttest_1samp(slopes, 0.0)
    return slopes, p_value


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

    for agent, dfs in aggregated.items():
        print(f"\n--- {agent} ---", file=f)

        slopes_p, p_p = regression_slope_test(dfs, "prob_correct")
        slopes_b, p_b = regression_slope_test(dfs, "brier")

        print(f"Probability slope mean: {np.mean(slopes_p):.4f}", file=f)
        print(f"Probability slope p-value: {p_p:.4g}", file=f)

        print(f"Brier slope mean: {np.mean(slopes_b):.4f}", file=f)
        print(f"Brier slope p-value: {p_b:.4g}", file=f)


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
