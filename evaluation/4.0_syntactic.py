"""
SYNTACTIC COMPLEXITY EVALUATION PIPELINE (REVISED)
-------------------------------------------------

Goal:
This pipeline evaluates whether LLM-generated dialogue deviates from the
original linguistic style of fictional characters at the syntactic level.
It uses dependency tree depth as an indicator of syntactic complexity and
quantifies (i) absolute deviation from a character-specific reference style
and (ii) systematic drift over dialogue turns.

Revisions relative to the initial version (as suggested by Polina):
- Baseline utterances are restricted to longer, non-fragmented utterances
  before random sampling.
- Two types of z-scores are computed:
    (a) reference-based z-scores relative to the baseline corpus;
    (b) simulation-based z-scores relative to the simulation corpus.
- Syntactic drift is estimated with pooled regression across runs instead of
  separate per-run regressions only.

-------------------------------------------------
Pipeline Steps
-------------------------------------------------

1) Reference Corpus Construction (Baseline)
   - Input: A character-labelled corpus derived from original literary texts.
   - Pre-filtering: For each character, we discard heavily fragmented or
     extremely short utterances based on a minimal length / depth criterion
     (e.g. at least K tokens or minimal dependency depth ≥ D).
   - From the remaining "long" utterances, we randomly sample up to 100
     utterances per character (fixed random seed for reproducibility).
   - Tool: pandas for data handling, spaCy for tokenization / parsing.
   - Purpose: Establish a cleaner, character-specific syntactic norm that is
     not dominated by artificially short fragments.

2) Dependency Parsing and Depth Extraction
   - Tool: spaCy (model: en_core_web_sm).
   - Each utterance is segmented into sentences.
   - For each sentence, we compute the maximal dependency tree depth using a
     recursive traversal from the root.
   - For each utterance, syntactic complexity is defined as the mean sentence
     depth (averaged across all sentences in that utterance).
   - Output: one syntactic depth value per utterance.

3) Reference Distribution Estimation (Baseline)
   - For each character, compute over the sampled baseline utterances:
       • mean dependency depth (μ_ref)
       • standard deviation of depth (σ_ref)
       • sample size (n_ref ≤ 100)
   - These statistics represent the character’s reference syntactic profile and
     serve as the target style for the LLM simulations.

4) Simulation Dialogue Processing
   - Input: LLM-generated multi-agent dialogues (Holmes, Poirot, Marple).
   - For each dialogue log, extract:
       • case ID
       • run ID
       • turn index
       • speaker
       • utterance text
   - Each utterance is parsed using the same spaCy pipeline as in the baseline.
   - For each utterance, compute syntactic depth as in Step 2.

5) Normalised Deviation Computation (Two z-score variants)
   For each simulated utterance with depth x_sim:

   (a) Reference-based z-score (primary OOC measure)
       z_ref = (x_sim − μ_ref) / σ_ref
       where μ_ref and σ_ref are the mean and SD of baseline depth for the
       corresponding character. This quantifies how unusual the utterance is
       with respect to the character’s original syntactic style.

   (b) Simulation-based z-score (exploratory measure)
       First, per character, compute:
           μ_sim, σ_sim
       over all simulated utterance depths.
       Then:
           z_sim = (x_sim − μ_sim) / σ_sim
       This standardises depths within the simulation corpus itself and is
       useful for describing internal variability.

   - To avoid numerical issues, a small epsilon is added if σ_ref or σ_sim is
     zero.
   - Output: per-turn depth, z_ref, and z_sim.

6) Per-Turn and Aggregate Statistics
   - Save per-turn syntactic depth and both z-scores to CSV.
   - For each character, compute:
       • reference_mean_depth (μ_ref) and reference_std_depth (σ_ref)
       • simulation_mean_depth (μ_sim) and simulation_std_depth (σ_sim)
       • depth_difference = μ_sim − μ_ref
       • relative_change (%) = depth_difference / μ_ref
       • mean_z_ref, std_z_ref
       • mean_z_sim, std_z_sim
       • number of evaluated turns (n_turns)

7) Pooled Syntactic Drift Analysis (Primary)
   - For each character separately, fit a pooled linear regression model across
     all cases and runs:
         dependent variable: z_ref (or z_sim as robustness check)
         independent variable: turn index
     Optionally, case/run IDs can be included as control variables.
   - Tool: scikit-learn (LinearRegression) or statsmodels (OLS).
   - Extract regression metrics:
       • slope (trend direction and magnitude)
       • R²
       • p-value
       • standard error
       • number of observations
   - Purpose: Test whether syntactic deviation systematically increases or
     decreases as the dialogue progresses, at the character level.

8) Optional Per-Run Drift Analysis
   - For comparability with earlier experiments, an optional per-(case, run,
     character) regression can be run:
         dependent variable: z_ref
         independent variable: turn index
       (only for runs with at least 5 turns).
   - Regression statistics are saved for descriptive analysis of run-level
     variability.

Interpretation:
- Larger absolute z_ref values indicate stronger deviation from a character’s
  reference syntactic style and thus potential out-of-character behaviour.
- Positive or negative slopes in the pooled regression suggest gradual
  syntactic drift over turns at the character level.
- z_sim and per-run regressions can be used as robustness and variability
  checks, while the main OOC conclusions are based on z_ref and pooled
  character-level trends.

"""

import pandas as pd
import numpy as np
import glob
import os
import random
import spacy
from sklearn.linear_model import LinearRegression
from scipy import stats

print("========== Syntactic Complexity Evaluation (Sentence-Level, normalised, revised) ==========")

# ----------------------------------------
# Step 0. Path & config settings
# ----------------------------------------
BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"

OUTPUT_DIR = "evaluation"
PER_TURN_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_summary.csv")
PER_RUN_REG_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_regression_per_run.csv")
POOLED_REG_FILE = os.path.join(OUTPUT_DIR, "4.0_syntactic_regression_pooled.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Baseline sampling configuration
REFERENCE_SAMPLE_SIZE = 100
RANDOM_SEED = 42

# Length filtering for baseline "long utterances"
MIN_TOKENS_BASELINE = 6   # you can tune this (e.g. 8, 10)
USE_DEPTH_FILTER = False  # set True if you prefer depth-based filtering
MIN_DEPTH_BASELINE = 2.0  # only used if USE_DEPTH_FILTER = True

# Regression options
MIN_TURNS_PER_RUN = 5
RUN_LEVEL_REGRESSION = True   # set False if you want to skip per-run regressions
USE_Z_REF_FOR_REG = True      # if False, pooled regressions will use z_sim instead


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
# Step 2. Utility functions: sentence-level dependency depth
# ----------------------------------------
def get_sentence_depth(sent):
    def depth(token):
        children = list(token.children)
        if not children:
            return 1
        return 1 + max(depth(child) for child in children)

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
    return float(np.mean(depths))


def count_tokens(text):
    doc = nlp(text)
    return len(doc)


# ----------------------------------------
# Step 3. Load baseline reference corpus
# ----------------------------------------
print("Loading baseline corpus...")

baseline_df = pd.read_csv(BASELINE_FILE)

baseline_df["character"] = (
    baseline_df["character"].astype(str).str.lower().str.strip()
)
baseline_df["quote"] = baseline_df["quote"].astype(str).str.strip()

characters = baseline_df["character"].unique()
print("Characters found in baseline:", characters)

# Pre-filter: keep only "long" utterances for each character
print("\nFiltering baseline utterances for minimal length...")

# Compute token length (or depth) once to avoid repeated parsing
baseline_df["token_len"] = baseline_df["quote"].apply(count_tokens)

if USE_DEPTH_FILTER:
    print("Using depth-based filtering with MIN_DEPTH_BASELINE =", MIN_DEPTH_BASELINE)
    baseline_df["depth"] = baseline_df["quote"].apply(get_utterance_depth)
else:
    print("Using token-length-based filtering with MIN_TOKENS_BASELINE =", MIN_TOKENS_BASELINE)

filtered_baseline_rows = []
for char in characters:
    char_df = baseline_df[baseline_df["character"] == char].copy()
    if USE_DEPTH_FILTER:
        char_df = char_df[char_df["depth"] >= MIN_DEPTH_BASELINE]
    else:
        char_df = char_df[char_df["token_len"] >= MIN_TOKENS_BASELINE]

    if len(char_df) == 0:
        print(f"[WARN] No baseline utterances left after filtering for character: {char}")
        continue
    filtered_baseline_rows.append(char_df)

baseline_df_filtered = pd.concat(filtered_baseline_rows, ignore_index=True)
print("Baseline size after filtering:", len(baseline_df_filtered))

# Fix random sampling for reproducibility (per character)
random.seed(RANDOM_SEED)
reference_stats = {}

print("\nComputing reference syntactic depth distributions from filtered baseline...")

baseline_samples = []
for char in baseline_df_filtered["character"].unique():
    char_df = baseline_df_filtered[baseline_df_filtered["character"] == char]
    char_quotes = char_df["quote"].tolist()

    sample_size = min(REFERENCE_SAMPLE_SIZE, len(char_quotes))
    sample_quotes = random.sample(char_quotes, sample_size)

    depths = [get_utterance_depth(q) for q in sample_quotes]
    mean_depth = float(np.mean(depths))
    std_depth = float(np.std(depths))

    # avoid division by zero in z-score
    if std_depth == 0:
        std_depth = 1e-6

    reference_stats[char] = {
        "mean_depth": mean_depth,
        "std_depth": std_depth,
        "n_samples": len(depths),
    }

    print(f"[{char}] mean={mean_depth:.3f}, std={std_depth:.3f}, n={len(depths)}")

    baseline_samples.extend(
        [
            {"character": char, "quote": q, "depth": d}
            for q, d in zip(sample_quotes, depths)
        ]
    )

baseline_sample_df = pd.DataFrame(baseline_samples)

# ----------------------------------------
# Step 4. Load simulation dialogues
# ----------------------------------------
dialogue_files = glob.glob(DATA_GLOB)
all_data = []

for file in dialogue_files:
    df = pd.read_csv(file)

    df["speaker"] = df["speaker"].astype(str).str.lower().str.strip()
    df["utterance"] = df["utterance"].astype(str).str.strip()

    run_dir = os.path.dirname(file)                # data/caseX/runY
    case_name = os.path.basename(os.path.dirname(run_dir))  # caseX
    run_id = os.path.basename(run_dir)             # runY

    df["run_id"] = run_id
    df["case"] = case_name

    all_data.append(df)

all_df = pd.concat(all_data, ignore_index=True)

print(f"\nTotal simulation utterances loaded: {len(all_df)}")
print("Speakers found:", all_df["speaker"].unique())

# ----------------------------------------
# Step 5. Compute syntactic depth for simulations
# ----------------------------------------
results = []

print("\nComputing syntactic depth for simulations and reference-based deviation...")

for _, row in all_df.iterrows():
    speaker = str(row["speaker"]).lower().strip()
    utterance = str(row["utterance"]).strip()
    turn = row["turn"]
    run_id = row["run_id"]
    case_name = row["case"]

    if speaker not in reference_stats:
        # unknown speaker, skip
        continue

    depth = get_utterance_depth(utterance)

    ref_mean = reference_stats[speaker]["mean_depth"]
    ref_std = reference_stats[speaker]["std_depth"]

    z_ref = (depth - ref_mean) / ref_std

    results.append(
        {
            "case": case_name,
            "run_id": run_id,
            "turn": turn,
            "character": speaker,
            "depth": depth,
            "reference_mean_depth": ref_mean,
            "reference_std_depth": ref_std,
            "z_ref": z_ref,
            "utterance": utterance,  # keep utterance text if you need it later
        }
    )

result_df = pd.DataFrame(results)

print("DEBUG: simulation depth mean:", result_df["depth"].mean())
print("DEBUG: simulation depth std:", result_df["depth"].std())

# ----------------------------------------
# Step 6. Compute simulation-based z-scores (z_sim)
# ----------------------------------------
print("\nComputing simulation-based z-scores per character...")

z_sim_values = []
for char, group in result_df.groupby("character"):
    depths = group["depth"].values
    sim_mean = float(np.mean(depths))
    sim_std = float(np.std(depths))
    if sim_std == 0:
        sim_std = 1e-6

    z_sim = (depths - sim_mean) / sim_std
    z_sim_values.append(pd.Series(z_sim, index=group.index))

# Concatenate and assign
if z_sim_values:
    z_sim_concat = pd.concat(z_sim_values)
    result_df["z_sim"] = z_sim_concat
else:
    result_df["z_sim"] = np.nan

# ----------------------------------------
# Step 7. Save per-turn results
# ----------------------------------------
result_df.to_csv(PER_TURN_FILE, index=False)

print("\n========== DONE (per-turn) ==========")
print(f"Saved syntactic results to: {PER_TURN_FILE}")
print(f"Total evaluated utterances: {len(result_df)}")

# ----------------------------------------
# Step 8. Summary statistics (reference vs simulation)
# ----------------------------------------
print("\n========== Summary Statistics (Reference vs Simulation) ==========")

summary_rows = []

for char in reference_stats.keys():
    char_df = result_df[result_df["character"] == char]

    if len(char_df) == 0:
        continue

    ref_mean = reference_stats[char]["mean_depth"]
    ref_std = reference_stats[char]["std_depth"]

    sim_mean = float(char_df["depth"].mean())
    sim_std = float(char_df["depth"].std())

    depth_diff = sim_mean - ref_mean
    relative_increase = depth_diff / ref_mean if ref_mean != 0 else 0.0

    mean_z_ref = float(char_df["z_ref"].mean())
    std_z_ref = float(char_df["z_ref"].std())

    mean_z_sim = float(char_df["z_sim"].mean())
    std_z_sim = float(char_df["z_sim"].std())

    summary_rows.append(
        {
            "character": char,
            "reference_mean_depth": ref_mean,
            "reference_std_depth": ref_std,
            "simulation_mean_depth": sim_mean,
            "simulation_std_depth": sim_std,
            "depth_difference": depth_diff,
            "relative_increase(%)": relative_increase * 100,
            "mean_z_ref": mean_z_ref,
            "std_z_ref": std_z_ref,
            "mean_z_sim": mean_z_sim,
            "std_z_sim": std_z_sim,
            "n_turns": len(char_df),
        }
    )

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
    turns=("turn", "nunique"),
).reset_index()

print(run_stats)

print("\nRuns with missing speakers:")
for run_id, df_run in all_df.groupby("run_id"):
    speakers = set(df_run["speaker"].unique())
    if speakers != {"holmes", "poirot", "marple"}:
        print(run_id, speakers)

# ----------------------------------------
# Step 9. Optional per-run Linear regression over turns (syntactic drift)
# ----------------------------------------
if RUN_LEVEL_REGRESSION:
    print("\n========== Linear Regression over Turns (Per Run) ==========")

    per_run_rows = []

    for (case_name, run_id, char), group_df in result_df.groupby(
        ["case", "run_id", "character"]
    ):
        if len(group_df) < MIN_TURNS_PER_RUN:
            continue

        X = group_df["turn"].values.reshape(-1, 1)
        y = group_df["z_ref"].values  # use reference-based z-score here

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r2 = model.score(X, y)

        slope_, intercept_, r_value, p_value, std_err = stats.linregress(
            group_df["turn"], y
        )

        n_turns = len(group_df)

        per_run_rows.append(
            {
                "case": case_name,
                "run_id": run_id,
                "character": char,
                "slope": slope_,
                "r2": r2,
                "p_value": p_value,
                "std_err": std_err,
                "n_turns": n_turns,
            }
        )

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(PER_RUN_REG_FILE, index=False)

    if not per_run_df.empty:
        print(per_run_df["case"].value_counts())
        print("Total per-run regression rows:", len(per_run_df))
        print(per_run_df.head(10))

    # Regression reliability by turn group (as before)
    print("\n========== Regression Reliability Analysis (Per Run) ==========")
    if not per_run_df.empty:
        reg_df = per_run_df.copy()
        reg_df["turn_group"] = reg_df["n_turns"].apply(
            lambda x: ">=5 turns" if x >= 5 else "3-4 turns"
        )

        reliability_summary = reg_df.groupby("turn_group").agg(
            mean_slope=("slope", "mean"),
            std_slope=("slope", "std"),
            n=("slope", "count"),
            mean_r2=("r2", "mean"),
        ).reset_index()

        print(reliability_summary)

# ----------------------------------------
# Step 10. Pooled regression across runs (primary drift analysis)
# ----------------------------------------
print("\n========== Pooled Regression across Runs (Per Character) ==========")

pooled_rows = []

z_col = "z_ref" if USE_Z_REF_FOR_REG else "z_sim"

for char, group_df in result_df.groupby("character"):
    # You can also impose a minimal total number of turns if needed
    if len(group_df) < MIN_TURNS_PER_RUN:
        continue

    X = group_df["turn"].values.reshape(-1, 1)
    y = group_df[z_col].values

    # Simple linear regression on turn only
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    r2 = model.score(X, y)

    slope_, intercept_, r_value, p_value, std_err = stats.linregress(
        group_df["turn"], y
    )

    pooled_rows.append(
        {
            "character": char,
            "z_variant": z_col,
            "slope": slope_,
            "r2": r2,
            "p_value": p_value,
            "std_err": std_err,
            "n_obs": len(group_df),
        }
    )

pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(POOLED_REG_FILE, index=False)

print(pooled_df)
print(f"\nSaved pooled regression results to: {POOLED_REG_FILE}")
