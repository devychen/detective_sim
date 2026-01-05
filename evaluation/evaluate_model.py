"""
evaluate_model.py

Purely analysis, no model training.

Evaluate character consistency (OOC) using a trained baseline classifier.

This script implements the three evaluation measures suggested by the advisor:

1. Classification confidence over dialogue turns
2. Brier score over dialogue turns
3. Linear regression analysis to detect systematic drift over time

The classifier is treated as an external reference model that estimates
how closely each utterance matches the canonical literary style of
Holmes, Marple, or Poirot.

默认：id2label = {
    0: "holmes",
    1: "marple",
    2: "poirot"
}
"""


# =================================================
# Imports
# =================================================

import os
import glob
import numpy as np
import pandas as pd

from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp

# =================================================
# Configuration (EDIT IF NEEDED)
# =================================================

MODEL_DIR = "./models/3class"
DIALOGUE_LOG_GLOB = "./data/case3/run_*/dialogue_log.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match training-time label mapping
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
    """
    Load the most recent checkpoint from a HuggingFace Trainer output directory.
    """
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
    """
    Load tokenizer and classifier model for inference only.
    """
    checkpoint = load_latest_checkpoint(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =================================================
# Core prediction function
# =================================================

def predict_probabilities(
    texts: List[str],
    tokenizer,
    model
) -> np.ndarray:
    """
    Predict class probabilities for a list of utterances.

    Returns
    -------
    probs : np.ndarray
        Shape (N, num_classes), softmax probabilities.
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
# Metric computations
# =================================================

def compute_brier_score(prob_correct: float) -> float:
    """
    Brier score for a single prediction.

    Brier = (p_correct - 1)^2
    """
    return (prob_correct - 1.0) ** 2


# =================================================
# Main evaluation logic
# =================================================

def evaluate_simulation(dialogue_path: str, tokenizer, model):
    """
    Evaluate one full simulation run.

    Returns
    -------
    results : dict
        {agent_name: DataFrame with per-turn metrics}
    """
    df = pd.read_csv(dialogue_path)

    results = {}

    for agent in ["Holmes", "Marple", "Poirot"]:
        agent_df = df[df["speaker"] == agent].copy()

        if agent_df.empty:
            continue

        texts = agent_df["utterance"].tolist()
        turns = agent_df["turn"].tolist()

        probs = predict_probabilities(texts, tokenizer, model)

        correct_label = agent.lower()
        correct_id = LABEL2ID[correct_label]

        prob_correct = probs[:, correct_id]
        brier = [compute_brier_score(p) for p in prob_correct]

        results[agent] = pd.DataFrame({
            "turn": turns,
            "prob_correct": prob_correct,
            "brier": brier
        })

    return results


# =================================================
# Aggregate over simulations
# =================================================

def aggregate_over_runs(all_runs: List[Dict[str, pd.DataFrame]]):
    """
    Combine multiple simulation runs into per-agent matrices.
    """
    aggregated = {a: [] for a in ["Holmes", "Marple", "Poirot"]}

    for run in all_runs:
        for agent, df in run.items():
            aggregated[agent].append(df)

    return aggregated


# =================================================
# Regression analysis
# =================================================

def regression_slope_test(dfs: List[pd.DataFrame], metric: str):
    """
    Fit per-run linear regressions and test slope significance.

    Parameters
    ----------
    dfs : list of DataFrames
    metric : "prob_correct" or "brier"

    Returns
    -------
    slopes : list of float
    p_value : float (one-sample t-test against zero)
    """
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
# Entry point
# =================================================

def main():
    tokenizer, model = load_classifier()

    dialogue_logs = sorted(glob.glob(DIALOGUE_LOG_GLOB))
    print(f"Found {len(dialogue_logs)} simulation runs.")

    all_runs = []

    for path in dialogue_logs:
        run_result = evaluate_simulation(path, tokenizer, model)
        all_runs.append(run_result)

    aggregated = aggregate_over_runs(all_runs)

    # -------------------------
    # Report results
    # -------------------------

    for agent, dfs in aggregated.items():
        print(f"\n=== {agent} ===")

        slopes_p, p_p = regression_slope_test(dfs, "prob_correct")
        slopes_b, p_b = regression_slope_test(dfs, "brier")

        print(f"Probability slope mean: {np.mean(slopes_p):.4f}")
        print(f"Probability slope p-value: {p_p:.4g}")

        print(f"Brier slope mean: {np.mean(slopes_b):.4f}")
        print(f"Brier slope p-value: {p_b:.4g}")


if __name__ == "__main__":
    main()
