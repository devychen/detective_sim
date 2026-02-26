"""
=========================================================
3.2_intra_agent_distance.py

INTRA-AGENT COSINE DISTANCE EVALUATION PIPELINE
-------------------------------------------------

This script evaluates whether lexical similarity within the same agent
reflects character-specific consistency rather than topic-driven similarity.

We define a corrected intra-agent lexical similarity metric that controls
for topic overlap across characters within the same dialogue turn.

Metric Definitions:

(1) Intra-agent similarity:
    Cosine similarity between consecutive utterances produced by the same character.

(2) Cross-agent similarity:
    Average cosine similarity between the current utterance of a character and
    the utterances produced by other characters at the same turn index.

(3) Character distance score (Intra-Agent Distance):
    character_distance = intra_agent_similarity - cross_agent_similarity

Interpretation:
- character_distance > 0:
    Lexical continuity is more likely attributable to character identity.
- character_distance ≈ 0:
    Lexical similarity may be driven by shared conversational topics.
- character_distance < 0:
    Lexical similarity across characters exceeds intra-character continuity,
    indicating potential out-of-character (OOC) behavior.

Important Note:
This metric is sensitive to the choice of text representation model.
In this implementation, TF-IDF embeddings are used to maintain consistency
with other lexical-level evaluation metrics in this study. Results should
therefore be interpreted with caution, especially when comparing across models.

Pipeline Steps:
1. Load dialogue logs from multiple simulation runs.
2. Normalize speaker labels and utterance texts.
3. Build a global TF-IDF vectorizer over all utterances.
4. Compute TF-IDF embeddings for each utterance.
5. For each run, compute:
   - intra-agent similarity between consecutive turns of the same character,
   - cross-agent similarity between characters at the same turn index.
6. Compute character distance scores for each utterance.
7. Aggregate per-character statistics and export results to CSV files.
8. Perform sanity checks for extreme similarity values.

Packages / Libraries:
- pandas, numpy: data loading and numerical computation
- scikit-learn: TF-IDF vectorization and cosine similarity computation
- glob, os: file system operations and batch processing utilities

=========================================================
"""


import pandas as pd
import numpy as np
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("========== Intra-Agent Distance Evaluation ==========")

# ----------------------------------------
# Step 0. Path settings
# ----------------------------------------
DATA_GLOB = "data/*/*/dialogue_log.csv"
OUTPUT_DIR = "evaluation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "3.2_intra_agent_distance.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Step 1. Load all dialogues
# ----------------------------------------
dialogue_files = glob.glob(DATA_GLOB)

all_data = []

for file in dialogue_files:
    df = pd.read_csv(file)

    # normalize speaker labels
    df["speaker"] = (
        df["speaker"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # normalize utterances
    df["utterance"] = (
        df["utterance"]
        .astype(str)
        .str.strip()
    )

    # Extract case and run_id from file path
    run_dir = os.path.dirname(file)                # data/caseX/runY
    case_name = os.path.basename(os.path.dirname(run_dir))  # caseX
    run_id = os.path.basename(run_dir)             # runY

    # add case and run_id (simulation identifier)
    df["case"] = case_name
    df["run_id"] = os.path.basename(os.path.dirname(file))

    all_data.append(df)

all_df = pd.concat(all_data, ignore_index=True)

print(f"Total utterances loaded: {len(all_df)}")
print("Speakers found:", all_df["speaker"].unique())
print("Cases found:", all_df["case"].unique())


# ----------------------------------------
# Step 2. Build global TF-IDF vectorizer
# ----------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(all_df["utterance"].tolist())

# store embeddings
embeddings = tfidf_matrix.toarray()

# ----------------------------------------
# Step 3. Compute intra-agent and cross-agent similarity
# ----------------------------------------
results = []

# group by run (each simulation independently)
for (case_name, run_id), run_df in all_df.groupby(["case", "run_id"]):
    run_df = run_df.sort_values(by=["turn", "speaker"]).reset_index(drop=True)

    # mapping: (turn, speaker) -> global index in all_df
    index_map = {}
    for _, row in run_df.iterrows():
        global_idx = all_df[
            (all_df["case"] == case_name) &
            (all_df["run_id"] == run_id) &
            (all_df["turn"] == row["turn"]) &
            (all_df["speaker"] == row["speaker"])
        ].index[0]

        index_map[(row["turn"], row["speaker"])] = global_idx

    for _, row in run_df.iterrows():
        turn = row["turn"]
        speaker = row["speaker"]

        # skip first turn (no previous utterance)
        if turn == 1:
            continue

        prev_key = (turn - 1, speaker)
        if prev_key not in index_map:
            continue

        idx_current = index_map[(turn, speaker)]
        idx_prev = index_map[prev_key]

        vec_current = embeddings[idx_current].reshape(1, -1)
        vec_prev = embeddings[idx_prev].reshape(1, -1)

        intra_sim = cosine_similarity(vec_current, vec_prev)[0][0]

        # ---- cross-agent similarity (same turn, other characters) ----
        cross_sims = []
        for other_speaker in run_df["speaker"].unique():
            if other_speaker == speaker:
                continue

            other_key = (turn, other_speaker)
            if other_key not in index_map:
                continue

            idx_other = index_map[other_key]
            vec_other = embeddings[idx_other].reshape(1, -1)

            sim_other = cosine_similarity(vec_current, vec_other)[0][0]
            cross_sims.append(sim_other)

        if len(cross_sims) == 0:
            continue

        cross_sim = np.mean(cross_sims)

        char_distance = intra_sim - cross_sim

        results.append({
            "case": case_name,
            "run_id": run_id,
            "turn": turn,
            "character": speaker,
            "intra_similarity": float(intra_sim),
            "cross_similarity": float(cross_sim),
            "character_distance": float(char_distance)
        })


# ----------------------------------------
# Step 4. Save results
# ----------------------------------------
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DONE ==========")
print(f"Saved results to: {OUTPUT_FILE}")
print(f"Total evaluated turns: {len(result_df)}")

# ----------------------------------------
# Step 5. Summary statistics
# ----------------------------------------
if len(result_df) > 0:
    print("\n========== Summary Statistics ==========")

    summary = result_df.groupby("character")[["intra_similarity", "cross_similarity", "character_distance"]].describe()
    print(summary)

    summary.to_csv(os.path.join(OUTPUT_DIR, "3.2_intra_agent_summary.csv"))
else:
    print("⚠️ No valid data for evaluation.")


# ----------------------------------------
# Step 6. Sanity check: extremely high similarity
# ----------------------------------------
print("\n========== Sanity Check: intra_similarity > 0.99 ==========")

high_sim_df = result_df[result_df["intra_similarity"] > 0.99][["run_id", "character", "turn", "intra_similarity"]]

print(f"Number of extremely high similarity cases: {len(high_sim_df)}")

if len(high_sim_df) > 0:
    print(high_sim_df.head(20))  # show first 20 cases
else:
    print("No extreme similarity cases found.")
