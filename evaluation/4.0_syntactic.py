"""
=========================================================
4.0_syntactic.py

SYNTACTIC COMPLEXITY EVALUATION PIPELINE
-------------------------------------------------

Goal:
This pipeline evaluates whether LLM-generated dialogue deviates from the original
linguistic style of fictional characters at the syntactic level. Specifically, it
measures changes in dependency tree depth as an indicator of syntactic complexity
and investigates whether syntactic drift occurs across dialogue turns.

Overview of Steps:

1) Reference Corpus Construction (Baseline)
   - Input: A character-labelled corpus derived from original literary texts.
   - For each character (Holmes, Poirot, Marple), 100 utterances are randomly sampled
     from the baseline corpus (fixed random seed for reproducibility).
   - Tool: pandas for data handling.
   - Purpose: Establish character-specific syntactic norms.

2) Dependency Parsing and Depth Extraction
   - Tool: spaCy (model: en_core_web_sm).
   - Each utterance is segmented into sentences.
   - For each sentence, the maximal dependency tree depth is computed recursively.
   - For each utterance, syntactic complexity is defined as the mean sentence depth.
   - Output: A syntactic depth value per utterance.

3) Reference Distribution Estimation
   - For each character, compute:
       • mean dependency depth
       • standard deviation (SD) of depth
       • sample size (n ≤ 100)
   - These statistics represent the character’s reference syntactic profile.

4) Simulation Dialogue Processing
   - Input: LLM-generated multi-agent dialogues (Holmes, Poirot, Marple).
   - Each utterance is parsed using the same spaCy pipeline.
   - For each utterance, dependency depth is computed.
   - Tool: glob, pandas.
   - Metadata extracted: case ID, run ID, turn number, speaker.

5) Normalised Deviation Computation (Z-score)
   - For each simulated utterance, compute:
       z = (depth_simulation − mean_reference) / SD_reference
   - Purpose: Quantify how far each utterance deviates from the character’s
     baseline syntactic style in standard deviation units.

6) Per-Turn and Aggregate Statistics
   - Save per-turn syntactic depth and z-scores.
   - Compute summary statistics per character:
       • mean depth (reference vs simulation)
       • depth difference and relative change (%)
       • mean and SD of z-scores
       • number of evaluated turns
   - Tool: pandas, numpy.

7) Syntactic Drift Analysis via Linear Regression
   - For each (case, run, character), fit a linear regression model:
       dependent variable: z-score
       independent variable: turn index
   - Tools: scikit-learn (LinearRegression), scipy (linregress).
   - Only runs with at least 5 turns are included to ensure minimal reliability.
   - Extract regression metrics:
       • slope (trend direction and magnitude)
       • R²
       • p-value
       • standard error
       • number of turns
   - Purpose: Detect systematic increase or decrease in syntactic deviation
     across dialogue progression.

8) Regression Reliability Analysis
   - Compare regression results across turn-length groups (e.g., ≥5 turns vs 3–4 turns). -- because i have done ≤ 3 and found 5 is better.
   - Evaluate stability of slope and explanatory power (R²).
   - Purpose: Assess whether syntactic drift signals are robust or artefacts
     of short dialogue sequences.

Interpretation:
- Larger absolute z-scores indicate stronger syntactic deviation from the character’s
  reference style, potentially signalling out-of-character (OOC) behaviour.
- Positive or negative regression slopes indicate gradual syntactic drift over turns.
- Together, these metrics provide a quantitative syntactic-level indicator of
  character consistency in multi-agent LLM simulations.

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

print("========== Syntactic Complexity Evaluation (Sentence-Level, normalised) ==========")

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

# Fix random sampling for reproducibility
baseline_df = baseline_df.groupby("character").sample(n=100, random_state=42)
print(f"Baseline sample size fixed to: {len(baseline_df)}")

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

    run_dir = os.path.dirname(file)               # data/caseX/runY
    case_name = os.path.basename(os.path.dirname(run_dir))  # caseX
    run_id = os.path.basename(run_dir)            # runY

    df["run_id"] = run_id
    df["case"] = case_name

    all_data.append(df)

all_df = pd.concat(all_data, ignore_index=True)

print(f"\nTotal simulation utterances loaded: {len(all_df)}")
print("Speakers found:", all_df["speaker"].unique())

# ----------------------------------------
# Step 6. Compute normalised syntactic deviation (z-score)
# ----------------------------------------
results = []

print("\nComputing syntactic depth and normalised deviation for simulations...")

for _, row in all_df.iterrows():
    speaker = row["speaker"]
    utterance = row["utterance"]
    turn = row["turn"]
    run_id = row["run_id"]
    case_name = row["case"]


    if speaker not in reference_stats:
        continue

    depth = get_utterance_depth(utterance)

    ref_mean = reference_stats[speaker]["mean_depth"]
    ref_std = reference_stats[speaker]["std_depth"]

    z_score = (depth - ref_mean) / ref_std

    results.append({
        "case": case_name,
        "run_id": run_id,
        "turn": turn,
        "character": speaker,
        "depth": depth,
        "reference_mean_depth": ref_mean,
        "reference_std_depth": ref_std,
        "z_score": z_score
    })

result_df = pd.DataFrame(results)

print("DEBUG: simulation depth mean:", result_df["depth"].mean())
print("DEBUG: simulation depth std:", result_df["depth"].std())


# ----------------------------------------
# Step 7. Save per-turn results
# ----------------------------------------
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DONE ==========")
print(f"Saved syntactic results to: {OUTPUT_FILE}")
print(f"Total evaluated utterances: {len(result_df)}")

# ----------------------------------------
# Step 8. Summary statistics (reference vs simulation)
# ----------------------------------------
print("\n========== Summary Statistics (Reference vs Simulation) ==========")

summary_rows = []

for char in characters:
    char_df = result_df[result_df["character"] == char]

    if len(char_df) == 0:
        continue

    ref_mean = reference_stats[char]["mean_depth"]
    ref_std = reference_stats[char]["std_depth"]

    sim_mean = char_df["depth"].mean()
    sim_std = char_df["depth"].std()

    mean_z = char_df["z_score"].mean()
    std_z = char_df["z_score"].std()

    depth_diff = sim_mean - ref_mean
    relative_increase = depth_diff / ref_mean if ref_mean != 0 else 0

    summary_rows.append({
        "character": char,
        "reference_mean_depth": ref_mean,
        "reference_std_depth": ref_std,
        "simulation_mean_depth": sim_mean,
        "simulation_std_depth": sim_std,
        "depth_difference": depth_diff,
        "relative_increase(%)": relative_increase * 100,
        "mean_z_score": mean_z,
        "std_z_score": std_z,
        "n_turns": len(char_df)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)
print(summary_df)

# ----------------------------------------
# DIAGNOSIS
# ----------------------------------------
print("\n========== RUN DIAGNOSTIC ==========")

run_stats = all_df.groupby("run_id").agg(
    total_utterances=("utterance", "count"),
    unique_speakers=("speaker", "nunique"),
    turns=("turn", "nunique")
).reset_index()

print(run_stats)

print("\nRuns with missing speakers:")
for run_id, df_run in all_df.groupby("run_id"):
    speakers = set(df_run["speaker"].unique())
    if speakers != {"holmes", "poirot", "marple"}:
        print(run_id, speakers)




# ----------------------------------------
# Step 9. Linear regression over turns (syntactic drift)
# ----------------------------------------
print("\n========== Linear Regression over Turns ==========")

regression_rows = []

for (case_name, run_id, char), group_df in result_df.groupby(["case", "run_id", "character"]):
    if len(group_df) < 5:
        continue


    X = group_df["turn"].values.reshape(-1, 1)
    y = group_df["z_score"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    r2 = model.score(X, y)

    slope_, intercept_, r_value, p_value, std_err = stats.linregress(group_df["turn"], y)

    n_turns = len(group_df)

    regression_rows.append({
        "case": case_name,
        "run_id": run_id,
        "character": char,
        "slope": slope_,
        "r2": r2,
        "p_value": p_value,
        "std_err": std_err,
        "n_turns": n_turns,
    })

regression_df = pd.DataFrame(regression_rows)
regression_df.to_csv(REGRESSION_FILE, index=False)

print(regression_df["case"].value_counts())
print("Total regression rows:", len(regression_df))

print(regression_df.head(100))
print(f"\nSaved regression results to: {REGRESSION_FILE}")


# ----------------------------------------
# REGRESSION RELIABILITY
# ----------------------------------------
print("\n========== Regression Reliability Analysis ==========")

reg_df = regression_df.copy() 

reg_df["turn_group"] = reg_df["n_turns"].apply(
    lambda x: ">=5 turns" if x >= 5 else "3-4 turns"
)

summary = reg_df.groupby("turn_group").agg(
    mean_slope=("slope", "mean"),
    std_slope=("slope", "std"),
    n=("slope", "count"),
    mean_r2=("r2", "mean")
).reset_index()

print(summary)


# syntactic drift is weak but consistent.
#========== Regression Reliability Analysis ==========
#   turn_group  mean_slope  std_slope   n   mean_r2
# 0  3-4 turns    0.180665   0.250996  33  0.478203
# 1  >=5 turns    0.101777   0.156201  39  0.341609
# so was thinking about limited to 5.




# OR SHOULD I USE WEIGHTED MEAN SLOPE?

# weighted_mean_slope = np.average(reg_df["slope"], weights=reg_df["n_turns"])
# print("\nWeighted mean slope:", weighted_mean_slope)

