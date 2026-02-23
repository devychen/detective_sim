"""
SYNTACTIC COMPLEXITY EVALUATION PIPELINE (REVISED)
--------------------------------------------------

增加了小数点后三位的精度控制

Goal:
    Evaluate whether LLM-generated dialogue deviates from the original
    syntactic style of fictional characters (Holmes, Poirot, Marple, etc.).
    Syntactic complexity is operationalised as dependency tree depth.
    The pipeline quantifies:
        (i) absolute deviation from a character-specific reference style,
        (ii) systematic syntactic drift over dialogue turns.


Revisions relative to the initial version:
    - Baseline utterances are restricted to longer, non-fragmented utterances
      before random sampling (token-length or depth-based filtering).
    - Two types of z-scores are computed:
        (a) reference-based z-scores (z_ref) relative to the baseline corpus:
            "How far does this utterance deviate from the original style?"
        (b) simulation-based z-scores (z_sim) within the simulation corpus:
            "Is this utterance extreme relative to other simulated utterances?"
    - Syntactic drift is estimated with pooled regression across runs
      (character-level) in addition to optional per-run regressions.


--------------------------------------------------
Pipeline Overview
--------------------------------------------------


0) Configuration
    - Baseline file: baseline/train_lines_clean_balanced_3class.csv
    - Simulation files: data/*/*/dialogue_log.csv (case/run structure)
    - Output directory: evaluation/
      * 4.0_syntactic_results.csv              (per-utterance results)
      * 4.0_syntactic_summary.csv             (per-character summary)
      * 4.0_syntactic_regression_per_run.csv  (optional run-level drift)
      * 4.0_syntactic_regression_pooled.csv   (pooled character-level drift)
    - Baseline sampling:
      * REFERENCE_SAMPLE_SIZE = 100
      * RANDOM_SEED = 42
    - Baseline filtering:
      * MIN_TOKENS_BASELINE (token-based, default)
        OR MIN_DEPTH_BASELINE if depth-based filtering is enabled.
    - Regression settings:
      * MIN_TURNS_PER_RUN = 5
      * RUN_LEVEL_REGRESSION = True / False
      * Pooled regressions are run per character for z_ref and z_sim.


1) Reference Corpus Construction (Baseline)
    - Input: character-labelled corpus of original literary dialogue
      (columns: character, quote).
    - Normalisation: lower-case and strip character names and quotes.
    - Pre-filtering:
      * Option A (default): compute token length via spaCy tokenisation and
        discard utterances shorter than MIN_TOKENS_BASELINE.
      * Option B: compute utterance-level depth and discard utterances with
        depth < MIN_DEPTH_BASELINE.
    - Per character, keep only the "long" utterances after filtering.
      Warn and skip if no utterances remain for a character.
    - Random sampling (per character):
      * Sample up to REFERENCE_SAMPLE_SIZE utterances using a fixed
        random seed for reproducibility.
      * For each sampled quote, compute utterance-level syntactic depth.
      * These depths define the reference distribution for that character.


2) Dependency Parsing and Depth Extraction
    - Tool: spaCy (model: en_core_web_sm).
    - Sentence-level depth:
      * For each sentence, find root tokens (token.head == token).
      * For each root, recursively traverse its dependency subtree.
      * The depth of a node = 1 + max(child depths); leaves have depth 1.
      * The sentence depth is the maximum depth across roots.
    - Utterance-level depth:
      * An utterance is parsed into sentences.
      * Compute the depth of each sentence as above.
      * Discard sentences with depth 0 (parsing anomalies).
      * Syntactic complexity for the utterance = mean sentence depth.
      * If no valid sentences are present, depth is set to 0.


3) Reference Distribution Estimation (Baseline)
    - For each character, using the sampled baseline utterances:
        * mean dependency depth:    μ_ref
        * standard deviation:       σ_ref
        * sample size:              n_ref (≤ REFERENCE_SAMPLE_SIZE)
    - If σ_ref == 0, replace with a small epsilon (1e-6) to avoid
      division-by-zero in z-score computations.
    - These statistics define the character-specific reference syntactic
      profile, serving as the target style for simulations.


4) Simulation Dialogue Processing
    - Input: LLM-generated multi-agent dialogues in CSV files with columns:
        * speaker, utterance, turn
      and implicit case/run information from the directory structure:
        * data/caseX/runY/dialogue_log.csv → case = caseX, run_id = runY
    - Normalisation:
        * Lower-case and strip speaker names.
        * Strip utterance text.
    - Concatenate all dialogue logs into a single DataFrame (all_df).
    - Speakers that do not appear in the reference_stats are ignored
      in subsequent steps.


5) Per-Utterance Depth and Reference-based z-score
    - For each simulation utterance:
        * If speaker is not in the reference_stats, skip this utterance.
        * Compute utterance depth using the same spaCy pipeline as the baseline.
        * Retrieve μ_ref and σ_ref for the corresponding character.
        * Compute reference-based z-score:
              z_ref = (depth - μ_ref) / σ_ref
    - Store per-utterance data:
        * case, run_id, turn, character, depth,
          reference_mean_depth, reference_std_depth, z_ref, utterance.


6) Simulation-based z-score (Internal Standardisation)
    - For each character, over all simulated utterances:
        * Compute μ_sim (mean depth) and σ_sim (std depth).
        * If σ_sim == 0, replace with epsilon (1e-6).
        * Compute simulation-based z-score for each utterance:
              z_sim = (depth - μ_sim) / σ_sim
    - Attach z_sim to the per-utterance results.


7) Per-Turn Results and Summary Statistics
    - Save per-utterance results to:
        * evaluation/4.0_syntactic_results.csv
          (contains depth, z_ref, z_sim and metadata per utterance).
    - For each character present in reference_stats:
        * Restrict to simulation utterances for that character.
        * Use μ_ref and σ_ref from the baseline (do not recompute).
        * Compute simulation-level statistics:
            - simulation_mean_depth, simulation_std_depth
            - depth_difference = simulation_mean_depth - reference_mean_depth
            - relative_increase(%) = (depth_difference / μ_ref) * 100
              (0 if μ_ref == 0)
            - mean_z_ref, std_z_ref
            - mean_z_sim, std_z_sim
            - n_turns (number of evaluated turns)
    - Save summary statistics to:
        * evaluation/4.0_syntactic_summary.csv


8) Diagnostics and Optional Per-Run Regression
    - Diagnostics:
        * Group all_df by run_id and compute:
            - total_utterances
            - unique_speakers
            - number of distinct turns
        * List runs where the set of speakers differs from the expected
          {holmes, poirot, marple} (for basic data quality checks).
    - Optional per-run syntactic drift (if RUN_LEVEL_REGRESSION is True):
        * For each (case, run_id, character) group in result_df:
            - Require at least MIN_TURNS_PER_RUN utterances.
            - Independent variable: turn index.
            - Dependent variable: z_ref (reference-based z-score).
            - Fit a linear regression (turn → z_ref) to obtain:
                slope, R², p-value, standard error, n_turns.
        * Save run-level regression results to:
            evaluation/4.0_syntactic_regression_per_run.csv
        * Optionally, compute a simple reliability summary by grouping
          runs into turn-count bins (e.g., >= 5 turns).


9) Pooled Syntactic Drift Analysis (Primary)
    - For each character, pool all simulation utterances across cases and runs.
    - Require at least MIN_TURNS_PER_RUN observations overall.
    - Run two regression variants per character:
        (1) Primary analysis:
            - Dependent variable: z_ref
            - Independent variable: turn index
            - Interpretation: drift relative to the canonical reference style.
        (2) Robustness analysis:
            - Dependent variable: z_sim
            - Independent variable: turn index
            - Interpretation: internal drift within the simulation corpus.
    - For each character × z_variant:
        * Fit a linear regression (turn → z_variant) and extract:
            - slope (direction and magnitude of drift)
            - R²
            - p-value
            - standard error
            - n_obs (number of observations)
    - Save pooled regression results to:
        * evaluation/4.0_syntactic_regression_pooled.csv


Interpretation:
    - Large absolute z_ref values indicate strong deviation from a
      character’s reference syntactic style and thus potential
      out-of-character behaviour.
    - Positive or negative slopes in pooled regressions suggest gradual
      syntactic drift over turns at the character level:
          * positive slope → increasing complexity relative to baseline,
          * negative slope → decreasing complexity relative to baseline.
    - z_sim and per-run regressions serve as robustness and variability
      checks, while the main OOC conclusions are based on z_ref and
      pooled character-level trends.
"""



import pandas as pd
import numpy as np
import glob
import os
import random
import spacy
from sklearn.linear_model import LinearRegression
from scipy import stats

# Global display option: show floats with 3 decimals in console prints
pd.options.display.float_format = "{:.3f}".format


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


print("DEBUG: simulation depth mean:", f"{result_df['depth'].mean():.3f}")
print("DEBUG: simulation depth std:", f"{result_df['depth'].std():.3f}")


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
result_df_rounded = result_df.round(3)
result_df_rounded.to_csv(PER_TURN_FILE, index=False)


print("\n========== DONE (per-turn) ==========")
print(f"Saved syntactic results to: {PER_TURN_FILE}")
print(f"Total evaluated utterances: {len(result_df_rounded)}")


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
summary_df_rounded = summary_df.round(3)
summary_df_rounded.to_csv(SUMMARY_FILE, index=False)
print(summary_df_rounded)


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
    per_run_df_rounded = per_run_df.round(3)
    per_run_df_rounded.to_csv(PER_RUN_REG_FILE, index=False)

    if not per_run_df_rounded.empty:
        print(per_run_df_rounded["case"].value_counts())
        print("Total per-run regression rows:", len(per_run_df_rounded))
        print(per_run_df_rounded.head(10))

    # Regression reliability by turn group (as before)
    print("\n========== Regression Reliability Analysis (Per Run) ==========")
    if not per_run_df_rounded.empty:
        reg_df = per_run_df_rounded.copy()
        reg_df["turn_group"] = reg_df["n_turns"].apply(
            lambda x: ">=5 turns" if x >= 5 else "3-4 turns"
        )

        reliability_summary = reg_df.groupby("turn_group").agg(
            mean_slope=("slope", "mean"),
            std_slope=("slope", "std"),
            n=("slope", "count"),
            mean_r2=("r2", "mean"),
        ).reset_index()

        print(reliability_summary.round(3))


# ----------------------------------------
# Step 10. Pooled regression across runs (primary + robustness)
# ----------------------------------------
print("\n========== Pooled Regression across Runs (Per Character) ==========")


"""
This step estimates syntactic drift at the character level using pooled
linear regression across all cases and runs.

Two regression variants are computed:

(1) Primary analysis:
    dependent variable = z_ref
    Interpretation: drift relative to the canonical reference style.

(2) Robustness check:
    dependent variable = z_sim
    Interpretation: internal drift within the simulation corpus.

This allows us to test whether systematic temporal drift reflects deviation
from the literary baseline (z_ref) or merely increasing internal variability
(z_sim).
"""


pooled_rows = []


for char, group_df in result_df.groupby("character"):

    # Skip if too few total observations
    if len(group_df) < MIN_TURNS_PER_RUN:
        continue

    for z_col in ["z_ref", "z_sim"]:

        X = group_df["turn"].values.reshape(-1, 1)
        y = group_df[z_col].values

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
pooled_df_rounded = pooled_df.round(3)
pooled_df_rounded.to_csv(POOLED_REG_FILE, index=False)

print(pooled_df_rounded)
print(f"\nSaved pooled regression results to: {POOLED_REG_FILE}")



# ----------------------------------------
# Step 11. Visualisation of Pooled Syntactic Drift (z_ref only)
# ----------------------------------------
print("\n========== Generating Visualisations (Pooled Drift) ==========")


"""
This step generates diagnostic plots for syntactic drift.

For each character:
    - Scatter plot: turn vs z_ref
    - Pooled regression line
    - 95% confidence interval band
    - Annotated slope and p-value

Only z_ref is visualised, as it constitutes the primary OOC measure.
z_sim results are retained for robustness checks in tabular form.
"""


import matplotlib.pyplot as plt
from scipy import stats as stats_vis  # avoid shadowing


PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_syntactic")
os.makedirs(PLOT_DIR, exist_ok=True)


for char, group_df in result_df.groupby("character"):

    if len(group_df) < MIN_TURNS_PER_RUN:
        continue

    x = group_df["turn"].values
    y = group_df["z_ref"].values

    # Fit regression
    slope, intercept, r_value, p_value, std_err = stats_vis.linregress(x, y)

    # Predicted values
    x_line = np.linspace(min(x), max(x), 100)
    y_line = intercept + slope * x_line

    # Compute 95% CI for regression line
    n = len(x)
    mean_x = np.mean(x)
    t_val = stats_vis.t.ppf(0.975, df=n - 2)
    s_err = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2))

    conf = t_val * s_err * np.sqrt(
        1 / n + (x_line - mean_x) ** 2 / np.sum((x - mean_x) ** 2)
    )

    lower = y_line - conf
    upper = y_line + conf

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.4)
    plt.plot(x_line, y_line)
    plt.fill_between(x_line, lower, upper, alpha=0.2)

    plt.xlabel("Turn")
    plt.ylabel("Reference-based z-score (z_ref)")
    plt.title(f"Syntactic Drift over Turns ({char.capitalize()})")

    # Annotate regression statistics
    plt.text(
        0.05,
        0.95,
        f"Slope = {slope:.3f}\np = {p_value:.3e}\nR² = {r_value**2:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )

    filename = os.path.join(PLOT_DIR, f"4.0_syntactic_drift_{char}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved plot: {filename}")
