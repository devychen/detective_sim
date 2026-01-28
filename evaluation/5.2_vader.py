"""
=========================================================
5.2_vader.py

PIPELINE OVERVIEW
-----------------
This script evaluates whether LLM agents remain in-character (IC)
at the discourse level using VADER sentiment analysis.

For each agent (Holmes, Poirot, Marple), we measure how their
utterances deviate from reference persona sentiment derived
from original literature texts.

Metrics computed:
1. Sentiment distance: Euclidean distance between turn-level
   sentiment vector and character-specific reference vector.
2. Sentiment trend: Linear regression slope of sentiment distance
   across turns, indicating persona drift.

Pipeline stages:

---------------------------------------------------------
(1) Reference sentiment computation (baseline corpus)
---------------------------------------------------------
- Load character-specific corpus used for the classifier training.
- Compute average VADER sentiment vectors per character:
      [positive, negative, neutral, compound]
- Purpose: define a baseline persona sentiment vector for each character.

Libraries used:
- pandas, numpy, vaderSentiment

Output:
- CSV: 5.1_reference_sentiment.csv

---------------------------------------------------------
(2) Turn-level sentiment computation
---------------------------------------------------------
- Load dialogue logs for each simulation run.
- Compute VADER sentiment for each utterance.
- Aggregate utterances within the same turn and same character
  to obtain a turn-level sentiment vector.
- Purpose: capture emotional content per character per turn.

Libraries used:
- pandas.groupby, numpy

Output:
- CSV: 5.1_turn_sentiment.csv

---------------------------------------------------------
(3) Sentiment distance computation
---------------------------------------------------------
- Compute Euclidean distance between each turn-level sentiment vector
  and the corresponding character baseline.
- Purpose: quantify how far the agent's utterance deviates from canonical persona.

Libraries used:
- numpy.linalg.norm

Output:
- CSV: 5.1_sentiment_distance.csv

---------------------------------------------------------
(4) Sentiment trend analysis (linear regression)
---------------------------------------------------------
- For each file and character:
    - Sort turns chronologically
    - Fit linear regression: distance ~ turn
    - Extract slope, intercept, R², and mean distance
- Purpose: measure persona drift over dialogue.

Libraries used:
- sklearn.linear_model.LinearRegression

Output:
- CSV: 5.1_sentiment_trend.csv

---------------------------------------------------------
(5) Summary aggregation (optional)
---------------------------------------------------------
- Aggregate by character, case, or case × character
  to compute mean and std of slope (drift) and distance (deviation)
- Purpose: enable comparative analysis of persona stability
  across agents and tasks.

Outputs:
- CSV: 5.2_summary_by_character.csv
- CSV: 5.2_summary_by_case.csv
- CSV: 5.2_summary_by_case_character.csv

---------------------------------------------------------
Packages / Libraries Used
---------------------------------------------------------
- vaderSentiment: sentiment analysis
- pandas, numpy: data processing and vector computation
- sklearn.linear_model: linear regression
- tqdm: progress bars for multiple dialogues
=========================================================
"""


import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# Step 0. Path settings
# ----------------------------------------

BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
MODEL_DIR = "./models/3class"   # not used here but kept for consistency
OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_REF_FILE = os.path.join(OUTPUT_DIR, "5.2_reference_sentiment.csv")
OUTPUT_TURN_FILE = os.path.join(OUTPUT_DIR, "5.2_turn_sentiment.csv")
OUTPUT_DIST_FILE = os.path.join(OUTPUT_DIR, "5.2_sentiment_distance.csv")
OUTPUT_TREND_FILE = os.path.join(OUTPUT_DIR, "5.2_sentiment_trend.csv")

analyzer = SentimentIntensityAnalyzer()

# ----------------------------------------
# Step 1. Compute reference sentiment vectors (baseline corpus)
# ----------------------------------------

def compute_vader_scores(text):
    s = analyzer.polarity_scores(str(text))
    return np.array([s["pos"], s["neg"], s["neu"], s["compound"]])

print("Loading baseline corpus...")
df_base = pd.read_csv(BASELINE_FILE)

# normalize character names (IMPORTANT)
df_base["character"] = df_base["character"].str.lower().str.strip()

ref_vectors = {}

for char, group in df_base.groupby("character"):
    scores = np.vstack(group["quote"].apply(compute_vader_scores))
    ref_vec = scores.mean(axis=0)
    ref_vectors[char] = ref_vec

df_ref = pd.DataFrame.from_dict(
    ref_vectors, orient="index",
    columns=["pos", "neg", "neu", "compound"]
)
df_ref.index.name = "character"
df_ref.to_csv(OUTPUT_REF_FILE)

print("Reference sentiment vectors saved:", OUTPUT_REF_FILE)

# ----------------------------------------
# Step 2. Compute turn-level sentiment vectors (robust schema)
# ----------------------------------------

turn_records = []

files = glob.glob(DATA_GLOB)

for file_path in tqdm(files, desc="Processing dialogues"):
    df = pd.read_csv(file_path)

    # ---- column mapping (adapt to your csv schema) ----
    COL_TURN = "turn"
    COL_CHAR = "speaker"
    COL_TEXT = "utterance"

    required_cols = {COL_TURN, COL_CHAR, COL_TEXT}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {file_path}: {required_cols - set(df.columns)}")

    # normalize character names (IMPORTANT)
    df[COL_CHAR] = df[COL_CHAR].str.lower().str.strip()

    for (turn_id, char), group in df.groupby([COL_TURN, COL_CHAR]):
        scores = np.vstack(group[COL_TEXT].apply(compute_vader_scores))
        turn_vec = scores.mean(axis=0)

        turn_records.append({
            "file": file_path,
            "turn_id": turn_id,
            "character": char,
            "pos": turn_vec[0],
            "neg": turn_vec[1],
            "neu": turn_vec[2],
            "compound": turn_vec[3]
        })

df_turn = pd.DataFrame(turn_records)
df_turn.to_csv(OUTPUT_TURN_FILE, index=False)

print("Turn-level sentiment saved:", OUTPUT_TURN_FILE)

# ----------------------------------------
# Step 3. Compute distance to reference vector
# ----------------------------------------

dist_records = []

for _, row in df_turn.iterrows():
    char = row["character"]

    if char not in ref_vectors:
        print(f"[WARN] Character not in baseline: {char}")
        continue

    turn_vec = np.array([row["pos"], row["neg"], row["neu"], row["compound"]])
    ref_vec = ref_vectors[char]

    dist = np.linalg.norm(turn_vec - ref_vec)

    dist_records.append({
        "file": row["file"],
        "turn_id": row["turn_id"],
        "character": char,
        "distance": dist
    })

df_dist = pd.DataFrame(dist_records)
df_dist.to_csv(OUTPUT_DIST_FILE, index=False)

print("Sentiment distance saved:", OUTPUT_DIST_FILE)

# ----------------------------------------
# Step 4. Trend analysis (linear regression)
# ----------------------------------------

trend_records = []

for (file_path, char), group in df_dist.groupby(["file", "character"]):
    group = group.sort_values("turn_id")

    if len(group) < 2:
        continue

    X = group["turn_id"].values.reshape(-1, 1)
    y = group["distance"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    trend_records.append({
        "file": file_path,
        "character": char,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mean_distance": y.mean()
    })

df_trend = pd.DataFrame(trend_records)
df_trend.to_csv(OUTPUT_TREND_FILE, index=False)

print("Sentiment trend analysis saved:", OUTPUT_TREND_FILE)

# ----------------------------------------
# Step 5. Summary statistics (NO new file needed)
# ----------------------------------------

print("\n===== Generating summary tables =====")

df = df_trend.copy()

# extract case name from file path: data/case1/run_xxx/dialogue_log.csv
df["case"] = df["file"].apply(lambda x: x.split("/")[1])

# ----------------------------------------
# 5.1 Character-level summary
# ----------------------------------------

char_summary = df.groupby("character").agg(
    mean_slope=("slope", "mean"),
    std_slope=("slope", "std"),
    mean_distance=("mean_distance", "mean"),
    std_distance=("mean_distance", "std")
).reset_index()

CHAR_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "5.2_summary_by_character.csv")
char_summary.to_csv(CHAR_SUMMARY_FILE, index=False)

print("Saved:", CHAR_SUMMARY_FILE)
print(char_summary)

# ----------------------------------------
# 5.2 Case-level summary
# ----------------------------------------

case_summary = df.groupby("case").agg(
    mean_slope=("slope", "mean"),
    mean_distance=("mean_distance", "mean"),
    std_slope=("slope", "std"),
    std_distance=("mean_distance", "std")
).reset_index()

CASE_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "5.2_summary_by_case.csv")
case_summary.to_csv(CASE_SUMMARY_FILE, index=False)

print("Saved:", CASE_SUMMARY_FILE)
print(case_summary)

# ----------------------------------------
# 5.3 Case + Character summary
# ----------------------------------------

case_char_summary = df.groupby(["case", "character"]).agg(
    mean_slope=("slope", "mean"),
    mean_distance=("mean_distance", "mean")
).reset_index()

CASE_CHAR_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "5.2_summary_by_case_character.csv")
case_char_summary.to_csv(CASE_CHAR_SUMMARY_FILE, index=False)

print("Saved:", CASE_CHAR_SUMMARY_FILE)
print(case_char_summary)

print("\n===== Summary generation finished =====")
