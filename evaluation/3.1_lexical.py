"""
=========================================================
3.1_lexical.py 

PIPELINE OVERVIEW
-----------------
This script evaluates character-specific lexical consistency in LLM-generated dialogues
using a TF-IDF-based similarity framework.

For each generated utterance, we compute its lexical similarity to the corresponding
character’s gold-standard corpus (Holmes, Poirot, Marple) by measuring cosine similarity
between TF-IDF representations.

The evaluation focuses on quantifying whether LLM agents preserve character-specific
vocabulary patterns or exhibit out-of-character (OOC) lexical behavior.

The evaluation consists of three main stages:

---------------------------------------------------------
(1) Character-specific TF-IDF modeling
---------------------------------------------------------
- Load gold-standard literary corpora for each character.
- Normalize character labels and preprocess text.
- Build independent TF-IDF vectorizers for each character.
- Compute centroid TF-IDF vectors representing each character’s lexical profile.
- Purpose: construct character-specific lexical reference spaces.

Methods / Libraries:
- scikit-learn TfidfVectorizer
- numpy, pandas

---------------------------------------------------------
(2) Utterance-level lexical similarity computation
---------------------------------------------------------
- Load LLM-generated dialogue logs from multiple simulation runs.
- For each utterance:
    * transform the text into a TF-IDF vector using the corresponding character model.
    * compute cosine similarity between the utterance vector and the character centroid.
- Aggregate similarity scores across turns, characters, and runs.
- Purpose: quantify lexical alignment between generated dialogue and original character style.

Methods / Libraries:
- scikit-learn cosine_similarity
- pandas, glob, os

---------------------------------------------------------
(3) Statistical analysis of lexical consistency
---------------------------------------------------------
- Compute descriptive statistics of similarity distributions per character.
- Conduct non-parametric significance testing:
    * Kruskal–Wallis test for overall group differences across characters.
    * Pairwise Mann–Whitney U tests with Bonferroni correction.
- Purpose: determine whether observed lexical differences across characters
  are statistically significant rather than random variation.

Methods / Libraries:
- scipy.stats (kruskal, mannwhitneyu)

---------------------------------------------------------
(4) Outputs
---------------------------------------------------------
- CSV files:
    * per-utterance lexical similarity scores
    * character-level summary statistics
    * statistical significance test results
- Console logs:
    * descriptive statistics and hypothesis testing results

---------------------------------------------------------
Packages / Libraries Used
---------------------------------------------------------
- pandas, numpy: data processing and numerical computation
- scikit-learn: TF-IDF vectorization and similarity computation
- scipy: statistical hypothesis testing
- glob, os: file system operations and batch processing

=========================================================

To align with the character-specific vocab rate, i use td-idf embbeding here, 
but this is might be too topic-sensitive.
Should I use others? What would be the recommended options?

"""



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

    if len(char_texts) == 0:
        print(f"⚠️ Warning: No gold text found for character: {char}")
        continue

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(char_texts)

    # convert np.matrix -> np.ndarray
    char_avg_vector = np.asarray(tfidf_matrix.mean(axis=0))

    vectorizers[char] = vectorizer
    character_vectors[char] = char_avg_vector

# ----------------------------------------
# Step 3. Compute lexical similarity
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

        if utterance == "" or utterance.lower() == "nan":
            skipped += 1
            continue

        if char not in vectorizers:
            print(f"⚠️ Unknown speaker skipped: {char}")
            skipped += 1
            continue

        vectorizer = vectorizers[char]
        char_vec = character_vectors[char]

        utter_vec = vectorizer.transform([utterance]).toarray()
        char_vec = np.asarray(char_vec)

        sim = cosine_similarity(utter_vec, char_vec)[0][0]

        results.append({
            "file": os.path.basename(os.path.dirname(file)),
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
# Step 5. Summary statistics
# ----------------------------------------
if len(sim_df) > 0:
    print("\n========== Summary Statistics ==========")
    summary = sim_df.groupby("character")["similarity"].describe()
    print(summary)

    summary.to_csv(os.path.join(OUTPUT_DIR, "3.1_lexical_summary_stats.csv"))
else:
    print("⚠️ No valid utterances were evaluated.")

# ----------------------------------------
# Step 6. Statistical significance tests
# ----------------------------------------
from scipy.stats import kruskal, mannwhitneyu

print("\n========== Statistical Significance Tests ==========")

# extract similarity scores by character
holmes_scores = sim_df[sim_df["character"] == "holmes"]["similarity"].values
marple_scores = sim_df[sim_df["character"] == "marple"]["similarity"].values
poirot_scores = sim_df[sim_df["character"] == "poirot"]["similarity"].values

# ---- Kruskal-Wallis test (3 groups) ----
H_stat, p_kw = kruskal(holmes_scores, marple_scores, poirot_scores)
print(f"Kruskal-Wallis test: H = {H_stat:.4f}, p = {p_kw:.6f}")

# ---- Pairwise Mann-Whitney U tests ----
pairs = [
    ("holmes", "marple", holmes_scores, marple_scores),
    ("holmes", "poirot", holmes_scores, poirot_scores),
    ("marple", "poirot", marple_scores, poirot_scores),
]

print("\nPairwise Mann-Whitney U tests (Bonferroni corrected):")

alpha = 0.05
bonferroni_alpha = alpha / len(pairs)

pairwise_results = []

for name1, name2, s1, s2 in pairs:
    U_stat, p_val = mannwhitneyu(s1, s2, alternative="two-sided")
    significant = p_val < bonferroni_alpha
    
    print(f"{name1} vs {name2}: U = {U_stat:.2f}, p = {p_val:.6f}, significant = {significant}")
    
    pairwise_results.append({
        "pair": f"{name1} vs {name2}",
        "U_stat": U_stat,
        "p_value": p_val,
        "significant_after_bonferroni": significant
    })

# save statistical test results
stat_df = pd.DataFrame(pairwise_results)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "3.1_lexical_significance_tests.csv"), index=False)

