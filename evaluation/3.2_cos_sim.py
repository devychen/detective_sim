'''

PIPELINE OVERVIEW
-----------------
This script evaluates intent distribution consistency between baseline (gold) data 
and LLM-generated dialogue simulations using a transformer-based intent classification model.
It quantifies how closely simulated dialogues preserve each character’s conversational intent profile 
compared to the original corpus, based on KL divergence of intent distributions.

The evaluation pipeline consists of six key stages:

---------------------------------------------------------
(1) Path and output setup
---------------------------------------------------------
- Define input paths for baseline and simulated dialogue data.
- Create output directories for storing intermediate and final results.
- Purpose: centralize file management and ensure reproducible data organization.
  Methods / Libraries: os, glob

---------------------------------------------------------
(2) Intent classification model loading
---------------------------------------------------------
- Load a pretrained intent classification model ("Falconsai/intent_classification") 
  using Hugging Face transformers.
- Define a helper function to predict intent labels for given text inputs.
- Purpose: automatically annotate each utterance or line with an intent category.
  Methods / Libraries: transformers.pipeline

---------------------------------------------------------
(3) Baseline corpus intent annotation
---------------------------------------------------------
- Load the human-authored baseline dataset of character dialogues (Holmes, Poirot, Marple).
- Apply the intent classifier to each line, computing corresponding intent labels.
- Save annotated data for downstream comparison.
  Methods / Libraries: pandas, tqdm

---------------------------------------------------------
(4) Dialogue simulation intent annotation
---------------------------------------------------------
- Load all generated dialogue logs from multiple simulation runs.
- Predict intents for each utterance.
- Combine all annotated simulations into a single DataFrame and export to CSV.
  Methods / Libraries: pandas, glob, transformers, tqdm

---------------------------------------------------------
(5) Intent distribution computation
---------------------------------------------------------
- For each character, compute normalized intent frequency distributions 
  for both baseline and generated dialogues.
- Aggregate them into a unified distribution table for comparison.
  Methods / Libraries: collections.Counter, pandas, numpy

---------------------------------------------------------
(6) KL divergence analysis
---------------------------------------------------------
- Align intent label spaces between the two distributions.
- Compute Kullback–Leibler (KL) divergence per character to measure 
  deviation of simulated intent patterns from baseline intent patterns.
- Save summary results to CSV.
  Methods / Libraries: numpy, scipy.stats.entropy

---------------------------------------------------------
(7) Outputs
---------------------------------------------------------
- CSV files:
    * 5.1_baseline_intents.csv — baseline dataset annotated with intent labels
    * 5.1_dialogue_intents.csv — simulated dialogues annotated with intent labels
    * 5.1_intent_distribution.csv — intent distribution comparison across characters
    * 5.1_kl_divergence.csv — character-level KL divergence scores
- Console logs:
    * intermediate progress via tqdm
    * final save paths and confirmation messages

---------------------------------------------------------
Purpose summary: 
Quantitatively assess how faithfully generated dialogues preserve 
original characters’ intent distributions, providing a measure of 
pragmatic consistency in character-driven LLM dialogue generation.
---------------------------------------------------------

'''

import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from transformers import pipeline
from scipy.stats import entropy

# ----------------------------------------
# Step 0. Path settings
# ----------------------------------------
BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_BASELINE_INTENT = os.path.join(OUTPUT_DIR, "5.1_baseline_intents.csv")
OUT_DIALOGUE_INTENT = os.path.join(OUTPUT_DIR, "5.1_dialogue_intents.csv")
OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "5.1_intent_distribution.csv")
OUT_KL = os.path.join(OUTPUT_DIR, "5.1_kl_divergence.csv")

# ----------------------------------------
# Step 1. Load intent classifier
# ----------------------------------------
intent_clf = pipeline(
    "text-classification",
    model="Falconsai/intent_classification",
    top_k=1
)

def get_intent(text):
    """Predict intent label for a single sentence."""
    if not isinstance(text, str) or text.strip() == "":
        return "EMPTY"
    result = intent_clf(text)[0]
    return result["label"]

# ----------------------------------------
# Step 2. Load baseline corpus
# ----------------------------------------
baseline_df = pd.read_csv(BASELINE_FILE)

assert "speaker" in baseline_df.columns
assert "text" in baseline_df.columns

tqdm.pandas(desc="Baseline intent classification")
baseline_df["intent"] = baseline_df["text"].progress_apply(get_intent)

baseline_df.to_csv(OUT_BASELINE_INTENT, index=False)
print(f"[Saved] Baseline intents -> {OUT_BASELINE_INTENT}")

# ----------------------------------------
# Step 3. Load dialogue simulation data
# ----------------------------------------
dialogue_files = glob.glob(DATA_GLOB)

dialogue_data = []

for file in dialogue_files:
    df = pd.read_csv(file)
    assert "speaker" in df.columns
    assert "utterance" in df.columns

    tqdm.pandas(desc=f"Dialogue intent classification: {file}")
    df["intent"] = df["utterance"].progress_apply(get_intent)

    df["source_file"] = file
    dialogue_data.append(df)

dialogue_df = pd.concat(dialogue_data, ignore_index=True)

dialogue_df.to_csv(OUT_DIALOGUE_INTENT, index=False)
print(f"[Saved] Dialogue intents -> {OUT_DIALOGUE_INTENT}")

# ----------------------------------------
# Step 4. Compute intent distributions
# ----------------------------------------
def compute_distribution(df, speaker):
    sub = df[df["speaker"] == speaker]
    counter = Counter(sub["intent"])
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}

speakers = ["Holmes", "Poirot", "Marple"]

baseline_dist = {s: compute_distribution(baseline_df, s) for s in speakers}
dialogue_dist = {s: compute_distribution(dialogue_df, s) for s in speakers}

# 转成表格方便分析
rows = []
for s in speakers:
    all_labels = sorted(set(baseline_dist[s].keys()) | set(dialogue_dist[s].keys()))
    for label in all_labels:
        rows.append({
            "speaker": s,
            "intent": label,
            "baseline_prob": baseline_dist[s].get(label, 0.0),
            "dialogue_prob": dialogue_dist[s].get(label, 0.0)
        })

dist_df = pd.DataFrame(rows)
dist_df.to_csv(OUT_DISTRIBUTION, index=False)
print(f"[Saved] Intent distribution -> {OUT_DISTRIBUTION}")

# ----------------------------------------
# Step 5. Align label space
# ----------------------------------------
def align_distributions(p, q, eps=1e-8):
    labels = sorted(set(p.keys()) | set(q.keys()))
    p_vec = np.array([p.get(l, eps) for l in labels])
    q_vec = np.array([q.get(l, eps) for l in labels])
    return labels, p_vec, q_vec

# ----------------------------------------
# Step 6. Compute KL divergence
# ----------------------------------------
kl_results = []

for s in speakers:
    labels, p_vec, q_vec = align_distributions(dialogue_dist[s], baseline_dist[s])
    kl = entropy(p_vec, q_vec)  # KL(P_dialogue || P_baseline)

    kl_results.append({
        "speaker": s,
        "KL_dialogue_vs_baseline": kl
    })

kl_df = pd.DataFrame(kl_results)
kl_df.to_csv(OUT_KL, index=False)
print(f"[Saved] KL divergence -> {OUT_KL}")
