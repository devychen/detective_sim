"""
=========================================================
4.0_syntactic.py

SYNTACTIC COMPLEXITY EVALUATION PIPELINE (NORMALIZED, SENTENCE-LEVEL)
-------------------------------------------------

This script evaluates syntactic deviation of each character in LLM simulations
relative to reference texts using dependency parsing.

Key improvements over naive syntactic metrics:

(1) Sentence-level dependency depth:
    - Each utterance is segmented into sentences.
    - Dependency tree depth is computed per sentence.
    - Utterance-level depth is aggregated as the mean sentence depth.
    This avoids bias caused by long multi-sentence LLM outputs.

(2) Normalized syntactic deviation (z-score):
    - Reference distributions are computed from baseline corpora.
    - For each utterance:
        z = (depth - reference_mean) / reference_std
    This allows comparison across characters with different syntactic baselines.

Pipeline Steps:
1. Load baseline corpus.
2. Sample reference utterances per character.
3. Compute sentence-level dependency depth statistics.
4. Load simulation dialogues.
5. Compute normalized syntactic deviation (z-score) per utterance.
6. Aggregate summary statistics.
7. Fit linear regression over turns to detect syntactic drift.
8. Save detailed results and summaries.

Packages / Libraries:
- pandas, numpy: data processing
- spacy: dependency parsing
- glob, os: file handling
- scipy, sklearn: regression and statistics

=========================================================
"""

import pandas as pd
import numpy as np
import glob
import os
import random
import spacy
from sklearn.linear_model import LinearRegression
from scipy import stats

print("========== Syntactic Complexity Evaluation (Sentence-Level, Normalized) ==========")

# ----------------------------------------
# Step 0. Path settings
# ----------------------------------------
BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"

OUTPUT_DIR = "evaluation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_summary.csv")
REGRESSION_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_regression.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Step 1. Load spaCy model
# ----------------------------------------
print("Loading spaCy model...")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Downloading en_core_web_sm...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ----------------------------------------
# Step 2. Utility function: sentence-level dependency depth
# ----------------------------------------
def get_sentence_depth(sent):
    def depth(token):
        if len(list(token.children)) == 0:
            return 1
        return 1 + max(depth(child) for child in token.children)

    roots = [token for token in sent if token.head == token]
    if len(roots) == 0:
        return 0

    return max(depth(root) for root in roots)


def get_utterance_depth(text):
    doc = nlp(text)
    depths = []

    for sent in doc.sents:
        d = get_sentence_depth(sent)
        if d > 0:
            depths.append(d)

    if len(depths) == 0:
        return 0

    # aggregation strategy: mean sentence depth
    return np.mean(depths)

# ----------------------------------------
# Step 3. Load baseline reference corpus
# ----------------------------------------
print("Loading baseline corpus...")

baseline_df = pd.read_csv(BASELINE_FILE)

baseline_df["character"] = baseline_df["character"].astype(str).str.lower().str.strip()
baseline_df["quote"] = baseline_df["quote"].astype(str).str.strip()

characters = baseline_df["character"].unique()
print("Characters found in baseline:", characters)

# ----------------------------------------
# Step 4. Sample reference utterances & compute depth statistics
# ----------------------------------------
REFERENCE_SAMPLE_SIZE = 100  # adjustable

reference_stats = {}

print("\nComputing reference syntactic depth distributions...")

for char in characters:
    char_quotes = baseline_df[baseline_df["character"] == char]["quote"].tolist()

    if len(char_quotes) == 0:
        continue

    sample_quotes = random.sample(char_quotes, min(REFERENCE_SAMPLE_SIZE, len(char_quotes)))

    depths = [get_utterance_depth(q) for q in sample_quotes]

    mean_depth = np.mean(depths)
    std_depth = np.std(depths)

    # avoid division by zero in z-score
    if std_depth == 0:
        std_depth = 1e-6

    reference_stats[char] = {
        "mean_depth": mean_depth,
        "std_depth": std_depth,
        "n_samples": len(depths)
    }

    print(f"[{char}] mean={mean_depth:.3f}, std={std_depth:.3f}, n={len(depths)}")

# ----------------------------------------
# Step 5. Load simulation dialogues
# ----------------------------------------
dialogue_files = glob.glob(DATA_GLOB)

all_data = []

for file in dialogue_files:
    df = pd.read_csv(file)

    df["speaker"] = df["speaker"].astype(str).str.lower().str.strip()
    df["utterance"] = df["utterance"].astype(str).str.strip()

    df["run_id"] = os.path.basename(os.path.dirname(file))

    all_data.append(df)

all_df = pd.concat(all_data, ignore_index=True)

print(f"\nTotal simulation utterances loaded: {len(all_df)}")
print("Speakers found:", all_df["speaker"].unique())

# ----------------------------------------
# Step 6. Compute normalized syntactic deviation (z-score)
# ----------------------------------------
results = []

print("\nComputing syntactic depth and normalized deviation for simulations...")

for _, row in all_df.iterrows():
    speaker = row["speaker"]
    utterance = row["utterance"]
    turn = row["turn"]
    run_id = row["run_id"]

    if speaker not in reference_stats:
        continue

    depth = get_utterance_depth(utterance)

    ref_mean = reference_stats[speaker]["mean_depth"]
    ref_std = reference_stats[speaker]["std_depth"]

    z_score = (depth - ref_mean) / ref_std

    results.append({
        "run_id": run_id,
        "turn": turn,
        "character": speaker,
        "depth": depth,
        "reference_mean_depth": ref_mean,
        "reference_std_depth": ref_std,
        "z_score": z_score
    })

result_df = pd.DataFrame(results)

# ----------------------------------------
# Step 7. Save per-turn results
# ----------------------------------------
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DONE ==========")
print(f"Saved syntactic results to: {OUTPUT_FILE}")
print(f"Total evaluated utterances: {len(result_df)}")

# ----------------------------------------
# Step 8. Summary statistics
# ----------------------------------------
print("\n========== Summary Statistics ==========")

summary_rows = []

for char in characters:
    char_df = result_df[result_df["character"] == char]

    if len(char_df) == 0:
        continue

    mean_z = char_df["z_score"].mean()
    std_z = char_df["z_score"].std()

    summary_rows.append({
        "character": char,
        "reference_mean_depth": reference_stats[char]["mean_depth"],
        "reference_std_depth": reference_stats[char]["std_depth"],
        "mean_z_score": mean_z,
        "std_z_score": std_z,
        "n_turns": len(char_df)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)
print(summary_df)

# ----------------------------------------
# Step 9. Linear regression over turns (syntactic drift)
# ----------------------------------------
print("\n========== Linear Regression over Turns ==========")

regression_rows = []

for (run_id, char), group_df in result_df.groupby(["run_id", "character"]):
    if len(group_df) < 3:
        continue

    X = group_df["turn"].values.reshape(-1, 1)
    y = group_df["z_score"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    r2 = model.score(X, y)

    slope_, intercept_, r_value, p_value, std_err = stats.linregress(group_df["turn"], y)

    regression_rows.append({
        "run_id": run_id,
        "character": char,
        "slope": slope_,
        "r2": r2,
        "p_value": p_value,
        "std_err": std_err
    })

regression_df = pd.DataFrame(regression_rows)
regression_df.to_csv(REGRESSION_FILE, index=False)

print(regression_df.head(20))
print(f"\nSaved regression results to: {REGRESSION_FILE}")
