"""
=========================================================
5.2_vader.py (REVISED)

增加visual，增加精度小数点后三位

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
- CSV: 5.2_reference_sentiment.csv


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
- CSV: 5.2_turn_sentiment.csv


---------------------------------------------------------
(3) Sentiment distance computation
---------------------------------------------------------
- Compute Euclidean distance between each turn-level sentiment vector
  and the corresponding character baseline.
- Purpose: quantify how far the agent's utterance deviates from canonical persona.

Libraries used:
- numpy.linalg.norm

Output:
- CSV: 5.2_sentiment_distance.csv


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
- CSV: 5.2_sentiment_trend.csv


---------------------------------------------------------
(5) Summary aggregation
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
(6) Optional visualisation (for thesis figures)
---------------------------------------------------------
- Controlled by VISUALIZE flag.
- If enabled, generate:
    (a) Turn vs distance plots with regression lines for each character
    (b) Boxplot of distance slopes by character

Outputs (if VISUALIZE=True):
- PNG: 5.2_distance_over_turns_<character>.png
- PNG: 5.2_slope_boxplot_by_character.png


Packages / Libraries Used
---------------------------------------------------------
- vaderSentiment: sentiment analysis
- pandas, numpy: data processing and vector computation
- sklearn.linear_model: linear regression
- tqdm: progress bars for multiple dialogues
- matplotlib, seaborn: optional visualisation
=========================================================
"""


import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# Optional visualisation flag
VISUALIZE = True

# Global display option: show floats with 3 decimals in console prints
try:
    pd.set_option('display.float_format', '{:.3f}'.format)
except Exception as e:
    print(f"Note: Could not set display float format: {e}")
    # 可选：使用备选方案
    # pd.options.display.float_format = '{:.3f}'.format


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

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_vader")
if VISUALIZE:
    os.makedirs(PLOT_DIR, exist_ok=True)


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
df_base["character"] = df_base["character"].astype(str).str.lower().str.strip()


ref_vectors = {}


for char, group in df_base.groupby("character"):
    scores = np.vstack(group["quote"].apply(compute_vader_scores))
    ref_vec = scores.mean(axis=0)
    ref_vectors[char] = ref_vec

    # brief reference vector with 3 decimals
    print(
        f"[REF] {char}: "
        f"pos={ref_vec[0]:.3f}, neg={ref_vec[1]:.3f}, "
        f"neu={ref_vec[2]:.3f}, compound={ref_vec[3]:.3f}"
    )


df_ref = pd.DataFrame.from_dict(
    ref_vectors, orient="index",
    columns=["pos", "neg", "neu", "compound"]
)
df_ref.index.name = "character"

df_ref_rounded = df_ref.round(3)
df_ref_rounded.to_csv(OUTPUT_REF_FILE)

print("Reference sentiment vectors saved:", OUTPUT_REF_FILE)


# ----------------------------------------
# Step 2. Compute turn-level sentiment vectors (robust schema)
# ----------------------------------------


turn_records = []


files = glob.glob(DATA_GLOB)


for file_path in tqdm(files, desc="Processing dialogues"):
    df = pd.read_csv(file_path)
    
    # Extract case and run_id from file path
    run_dir = os.path.dirname(file_path)                # data/caseX/runY
    case_name = os.path.basename(os.path.dirname(run_dir))  # caseX
    run_id = os.path.basename(run_dir)                  # runY

    # ---- column mapping (adapt to your csv schema) ----
    COL_TURN = "turn"
    COL_CHAR = "speaker"
    COL_TEXT = "utterance"

    required_cols = {COL_TURN, COL_CHAR, COL_TEXT}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {file_path}: {required_cols - set(df.columns)}")

    # normalize character names (IMPORTANT)
    df[COL_CHAR] = df[COL_CHAR].astype(str).str.lower().str.strip()

    for (turn_id, char), group in df.groupby([COL_TURN, COL_CHAR]):
        scores = np.vstack(group[COL_TEXT].apply(compute_vader_scores))
        turn_vec = scores.mean(axis=0)

        turn_records.append({
            "case": case_name,
            "run_id": run_id,
            "turn": turn_id,
            "character": char,
            "pos": turn_vec[0],
            "neg": turn_vec[1],
            "neu": turn_vec[2],
            "compound": turn_vec[3]
        })


df_turn = pd.DataFrame(turn_records)
df_turn_rounded = df_turn.round(3)
df_turn_rounded.to_csv(OUTPUT_TURN_FILE, index=False)

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

    dist = float(np.linalg.norm(turn_vec - ref_vec))

    dist_records.append({
        "case": row["case"],
        "run_id": row["run_id"],
        "turn": row["turn"],
        "character": char,
        "distance": dist
    })


df_dist = pd.DataFrame(dist_records)
df_dist_rounded = df_dist.round(3)
df_dist_rounded.to_csv(OUTPUT_DIST_FILE, index=False)

print("Sentiment distance saved:", OUTPUT_DIST_FILE)


# ----------------------------------------
# Step 4. Trend analysis (linear regression)
# ----------------------------------------


trend_records = []


for (case_name, run_id, char), group in df_dist.groupby(["case", "run_id", "character"]):
    group = group.sort_values("turn")

    if len(group) < 2:
        continue

    X = group["turn"].values.reshape(-1, 1)
    y = group["distance"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, y))

    trend_records.append({
        "case": case_name,
        "run_id": run_id,
        "turn": turn_id,
        "character": char,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mean_distance": float(y.mean())
    })


df_trend = pd.DataFrame(trend_records)
df_trend_rounded = df_trend.round(3)
df_trend_rounded.to_csv(OUTPUT_TREND_FILE, index=False)

print("Sentiment trend analysis saved:", OUTPUT_TREND_FILE)
print(df_trend_rounded.head())


# ----------------------------------------
# Step 5. Summary statistics
# ----------------------------------------


print("\n===== Generating summary tables =====")


df = df_trend.copy()


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
char_summary_rounded = char_summary.round(3)
char_summary_rounded.to_csv(CHAR_SUMMARY_FILE, index=False)

print("Saved:", CHAR_SUMMARY_FILE)
print(char_summary_rounded)


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
case_summary_rounded = case_summary.round(3)
case_summary_rounded.to_csv(CASE_SUMMARY_FILE, index=False)

print("Saved:", CASE_SUMMARY_FILE)
print(case_summary_rounded)


# ----------------------------------------
# 5.3 Case + Character summary
# ----------------------------------------


case_char_summary = df.groupby(["case", "character"]).agg(
    mean_slope=("slope", "mean"),
    mean_distance=("mean_distance", "mean")
).reset_index()

CASE_CHAR_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "5.2_summary_by_case_character.csv")
case_char_summary_rounded = case_char_summary.round(3)
case_char_summary_rounded.to_csv(CASE_CHAR_SUMMARY_FILE, index=False)

print("Saved:", CASE_CHAR_SUMMARY_FILE)
print(case_char_summary_rounded)


print("\n===== Summary generation finished =====")


# ----------------------------------------
# Step 6. Optional visualisation
# ----------------------------------------
if VISUALIZE:
    print("\n===== Generating VADER visualisations =====")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # 6.1 Turn vs distance with regression line (per character)
    for char, group in df_dist.groupby("character"):
        if len(group) < 2:
            continue

        group = group.sort_values("turn")

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=group, x="turn", y="distance", alpha=0.4)
        sns.regplot(
            data=group,
            x="turn",
            y="distance",
            scatter=False,
            color="C1",
            line_kws={"linewidth": 2},
        )

        plt.xlabel("Turn")
        plt.ylabel("Distance to baseline sentiment")
        plt.title(f"Sentiment Distance over Turns ({char})")

        fname = os.path.join(PLOT_DIR, f"5.2_distance_over_turns_{char}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

        print("Saved plot:", fname)

    # 6.2 Boxplot of slope by character
    if not df_trend.empty:
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=df_trend, x="character", y="slope")
        plt.xlabel("Character")
        plt.ylabel("Slope of distance over turns")
        plt.title("Sentiment Drift Slopes by Character")

        fname_box = os.path.join(PLOT_DIR, "5.2_slope_boxplot_by_character.png")
        plt.tight_layout()
        plt.savefig(fname_box, dpi=300)
        plt.close()

        print("Saved plot:", fname_box)

    print("===== Visualisation finished =====")
