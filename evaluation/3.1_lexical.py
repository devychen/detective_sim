import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os
import numpy as np

print("========== Lexical Similarity Evaluation ==========")

# ----------------------------------------
# Step 0. Path settings
# ----------------------------------------
GOLD_PATH = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
OUTPUT_DIR = "evaluation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "3.1_lexical_similarity.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Step 1. Load gold standard data
# ----------------------------------------
gold_df = pd.read_csv(GOLD_PATH)

# normalize character names
gold_df["character"] = gold_df["character"].astype(str).str.lower()
gold_df["quote"] = gold_df["quote"].astype(str)

characters = gold_df["character"].unique()
print("Finished building TF-IDF models for:", characters)

# ----------------------------------------
# Step 2. Build TF-IDF models for each character
# ----------------------------------------
vectorizers = {}
character_vectors = {}

for char in characters:
    char_texts = gold_df[gold_df["character"] == char]["quote"].tolist()

    # safety check
    if len(char_texts) == 0:
        print(f"⚠️ Warning: No gold text found for character: {char}")
        continue

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),     # unigram + bigram (better stylistic signal)
        min_df=2                # ignore extremely rare words
    )
    
    tfidf_matrix = vectorizer.fit_transform(char_texts)

    # average vector = character lexical profile
    char_avg_vector = tfidf_matrix.mean(axis=0)

    vectorizers[char] = vectorizer
    character_vectors[char] = char_avg_vector

# ----------------------------------------
# Step 3. Load dialogue logs and compute similarity
# ----------------------------------------
dialogue_files = glob.glob(DATA_GLOB)

results = []
skipped = 0

for file in dialogue_files:
    print("Processing:", file)
    df = pd.read_csv(file)

    for _, row in df.iterrows():
        char = str(row["speaker"]).lower().strip()
        utterance = str(row["utterance"]).strip()

        # skip empty utterances
        if utterance == "" or utterance.lower() == "nan":
            skipped += 1
            continue

        # skip unknown characters
        if char not in vectorizers:
            print(f"⚠️ Unknown speaker skipped: {char}")
            skipped += 1
            continue

        vectorizer = vectorizers[char]
        char_vec = character_vectors[char]

        # transform utterance into tf-idf vector
        utter_vec = vectorizer.transform([utterance])

        # cosine similarity
        sim = cosine_similarity(utter_vec, char_vec).flatten()[0]

        results.append({
            "file": os.path.basename(os.path.dirname(file)),  # run_xxx folder
            "turn": row["turn"],
            "character": char,
            "utterance": utterance,
            "similarity": float(sim)
        })

# ----------------------------------------
# Step 4. Save results
# ----------------------------------------
sim_df = pd.DataFrame(results)
sim_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DONE ==========")
print(f"Saved lexical similarity results to: {OUTPUT_FILE}")
print(f"Total evaluated utterances: {len(sim_df)}")
print(f"Total skipped utterances: {skipped}")

# ----------------------------------------
# Step 5. Basic statistics (for sanity check & paper)
# ----------------------------------------
if len(sim_df) > 0:
    print("\n========== Summary Statistics ==========")
    summary = sim_df.groupby("character")["similarity"].describe()
    print(summary)

    summary.to_csv(os.path.join(OUTPUT_DIR, "3.1_lexical_summary_stats.csv"))
else:
    print("⚠️ No valid utterances were evaluated. Check character names and data format.")
