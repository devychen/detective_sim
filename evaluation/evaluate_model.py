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
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from scipy.stats import linregress


# =================================================
# Configuration
# =================================================

MODEL_DIR = "models/3class"   # trained baseline classifier
DATA_ROOT = "data/case3"      # where simulation runs are stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================
# Helper functions
# =================================================

def load_latest_checkpoint(model_dir: str):
    """
    Load the most recent checkpoint from a HuggingFace Trainer directory.

    This mirrors the logic used during training and ensures we always
    evaluate with the latest available model.
    """
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in model directory.")

    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt


def load_classifier(model_dir: str):
    """
    Load tokenizer and classifier model.
    """
    checkpoint = load_latest_checkpoint(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def predict_probabilities(
    texts: List[str],
    tokenizer,
    model
) -> np.ndarray:
    """
    Predict class probabilities for a list of utterances.

    Returns
    -------
    np.ndarray
        Shape: (n_samples, n_classes)
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
        logits = outputs.logits.cpu().numpy()

    return softmax(logits, axis=1)


def compute_brier_score(p_correct: float) -> float:
    """
    Compute Brier score for a single prediction.

    Since the true label is known and binary (correct class vs others),
    the Brier score reduces to:

        (p_correct - 1)^2
    """
    return (p_correct - 1.0) ** 2


# =================================================
# Main evaluation logic
# =================================================

def evaluate_all_runs():
    """
    Evaluate OOC metrics across all simulation runs.

    Returns
    -------
    pd.DataFrame
        One row per utterance, with probability and Brier score.
    """
    tokenizer, model = load_classifier(MODEL_DIR)

    all_rows = []

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(DATA_ROOT, "run_*"))
        if os.path.isdir(d)
    )

    for run_dir in run_dirs:
        dialogue_path = os.path.join(run_dir, "dialogue_log.csv")
        df = pd.read_csv(dialogue_path)

        texts = df["utterance"].tolist()
        speakers = df["speaker"].tolist()
        turns = df["turn"].tolist()

        probs = predict_probabilities(texts, tokenizer, model)

        id2label = model.config.id2label
        label2id = {v: k for k, v in id2label.items()}

        for i in range(len(df)):
            speaker = speakers[i].lower()
            true_label_id = label2id[speaker]

            p_correct = probs[i, true_label_id]
            brier = compute_brier_score(p_correct)

            all_rows.append({
                "run": os.path.basename(run_dir),
                "turn": turns[i],
                "speaker": speaker,
                "p_correct": p_correct,
                "brier": brier
            })

    return pd.DataFrame(all_rows)


# =================================================
# Turn-level aggregation & regression
# =================================================

def analyze_trends(df: pd.DataFrame):
    """
    Analyze probability and Brier score trends over turns
    using linear regression.
    """
    results = []

    for speaker in df["speaker"].unique():
        sub = df[df["speaker"] == speaker]

        # aggregate over runs
        grouped = sub.groupby("turn").agg(
            mean_p=("p_correct", "mean"),
            mean_brier=("brier", "mean")
        ).reset_index()

        # linear regression
        p_reg = linregress(grouped["turn"], grouped["mean_p"])
        brier_reg = linregress(grouped["turn"], grouped["mean_brier"])

        results.append({
            "speaker": speaker,
            "p_slope": p_reg.slope,
            "p_pvalue": p_reg.pvalue,
            "brier_slope": brier_reg.slope,
            "brier_pvalue": brier_reg.pvalue
        })

        print("\n====================================")
        print(f"Speaker: {speaker.upper()}")
        print("Probability trend:")
        print(f"  slope = {p_reg.slope:.4f}, p = {p_reg.pvalue:.4f}")
        print("Brier score trend:")
        print(f"  slope = {brier_reg.slope:.4f}, p = {brier_reg.pvalue:.4f}")

    return pd.DataFrame(results)


# =================================================
# Entry point
# =================================================

if __name__ == "__main__":
    print("Evaluating character consistency (OOC)...")

    df = evaluate_all_runs()
    df.to_csv("baseline_ooc_scores.csv", index=False)

    print("\nSaved per-utterance scores to baseline_ooc_scores.csv")

    trend_df = analyze_trends(df)
    trend_df.to_csv("baseline_ooc_trends.csv", index=False)

    print("\nSaved trend analysis to baseline_ooc_trends.csv")
