"""
================================================================================
Clustering Analysis (Validation of Character Contamination in Simulation)
================================================================================

Purpose
-------
This script evaluates character separability in:

1) Reference corpus (ground-truth utterances)
2) Simulation dialogues (collaborative runs)

The goal is to test whether collaborative simulation introduces
"contamination" between character representations, i.e., whether
multi-agent collaboration reduces character-distinctive features.

Design Logic
------------
- Cluster utterances into 3 clusters (Holmes, Marple, Poirot) using KMeans.
- Perform clustering separately on:
    * Reference corpus (bootstrapped to match simulation size)
    * Simulation dialogues
- Evaluate clustering quality with respect to:
    * Character identity (ARI_character)
    * Case identity (ARI_case; only applicable for simulation)
- Compare metrics between reference and simulation:
    * Weaker separability (lower silhouette, purity, ARI_character) in simulation
      indicates potential representational contamination.

Vector Representations
----------------------
Two approaches:

1) Embedding vectors (default):
   - Use the encoder from the fine-tuned 3-class classifier.
   - Extract CLS token hidden states.
   - Reference corpus is bootstrapped to match the number of simulation utterances.
   - Metrics:
       * Silhouette Coefficient
       * Purity
       * ARI_character (always)
       * ARI_case (simulation only)

2) Metric vectors:
   - Concatenate turn-level validation metrics into a vector.
   - Mandatory metrics: lexical_similarity, character_distance, z_ref, depth, sentiment_distance
   - Optional metrics: z_sim, intra_similarity (controlled via INCLUDE_OPTIONAL_METRICS switch)
   - Metrics:
       * Silhouette Coefficient
       * Purity
       * ARI_character (always)
       * ARI_case (None; not meaningful for reference or metrics)

Clustering Procedure
-------------------
- Standardize features before clustering.
- KMeans parameters: n_clusters=3, n_init=50, random_state=42.
- Reference corpus: run N_BOOTSTRAP_REF bootstrap iterations to downsample
  to the simulation size (387 utterances).
- Simulation data: cluster using true case and character labels.

Evaluation Metrics
------------------
Mandatory:
- Silhouette Score: cluster cohesion and separation
- Purity: proportion of majority true labels in each cluster
- Adjusted Rand Index (ARI): agreement between clusters and true labels
  * ARI_character: alignment with character identity
  * ARI_case: alignment with case (simulation only)

Output
------
CSV: evaluation/7.1_clustering_results.csv
Columns:
data_type | vector_type | silhouette | purity | ARI_character | ARI_case

Notes
-----
- Reference corpus does not have case structure, so ARI_case is always None.
- Metric vector clustering can optionally include additional metrics via
  INCLUDE_OPTIONAL_METRICS.
- Randomness in bootstrap may slightly vary reference results, but simulation
  clustering is deterministic if random_state is fixed.
================================================================================
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch

# ==============================================================================
# User Settings
# ==============================================================================

GOLD_PATH = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
METRIC_PATH = "evaluation/validation_turn_level_aggregated.csv"
MODEL_DIR = "./models/3class/checkpoint-2748"
OUTPUT_DIR = "evaluation"

VECTOR_TYPE = "embedding"   # "embedding" or "metric"
INCLUDE_OPTIONAL_METRICS = True

N_BOOTSTRAP_REF = 5         # reference downsampling iterations
K_CLUSTERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_embeddings(texts, tokenizer, model, batch_size=32):
    """
    Step 1: Compute CLS embeddings for a list of texts
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(DEVICE)

            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


def load_simulation_data():
    files = glob.glob(DATA_GLOB)
    all_texts = []
    all_speakers = []
    all_cases = []

    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values(["turn"])

        # 从路径里提取 case 名
        # 假设路径格式: data/case1/runX/dialogue_log.csv
        case_name = f.split("/")[1]

        all_texts.extend(df["utterance"].tolist())
        all_speakers.extend(df["speaker"].tolist())
        all_cases.extend([case_name] * len(df))

    return all_texts, all_speakers, all_cases


def load_metric_vectors(data_type="simulation"):
    """
    Load metric-based vectors.

    Mandatory metrics:
        lexical_similarity
        character_distance
        z_ref
        depth
        sentiment_distance

    Optional:
        z_sim
        intra_similarity
    """

    df = pd.read_csv(METRIC_PATH)

    mandatory = [
        "lexical_similarity",
        "character_distance",
        "z_ref",
        "depth",
        "sentiment_distance"
    ]

    optional = [
        "z_sim",
        "intra_similarity"
    ]

    if INCLUDE_OPTIONAL_METRICS:
        cols = mandatory + optional
    else:
        cols = mandatory

    cols = [c for c in cols if c in df.columns]

    df = df.dropna(subset=cols)

    vectors = df[cols].values
    labels = df["character"].tolist()

    return vectors, labels


def compute_purity(y_true, clusters):
    """
    Compute cluster purity.
    """
    y_true = np.array(y_true)
    purity_sum = 0

    for k in range(K_CLUSTERS):
        cluster_indices = np.where(clusters == k)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_labels = y_true[cluster_indices]
        majority_count = np.max(np.bincount(
            pd.factorize(cluster_labels)[0]
        ))

        purity_sum += majority_count

    return purity_sum / len(y_true)


def evaluate_clustering(X, y_true):
    """
    Step 2: Standardize features
    Step 3: KMeans clustering
    Step 4: Compute evaluation metrics
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=50,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)
    purity = compute_purity(y_true, clusters)
    ari = adjusted_rand_score(y_true, clusters)

    return sil, purity, ari


# ==============================================================================
# Main Procedure
# ==============================================================================

results = []

# ------------------------------------------------------------------------------
# EMBEDDING-BASED CLUSTERING
# ------------------------------------------------------------------------------

if VECTOR_TYPE == "embedding":

    print("Loading fine-tuned model from:", MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # ---------------------------
    # Reference Corpus (Bootstrap)
    # ---------------------------

    df_ref_full = pd.read_csv(GOLD_PATH)

    print("Running bootstrap clustering on reference corpus...")

    for i in range(N_BOOTSTRAP_REF):

        # Step 1: Downsample reference to match simulation size (387)
        df_sample = df_ref_full.sample(
            n=387,
            random_state=42 + i
        )

        texts = df_sample["quote"].tolist()
        labels = df_sample["character"].tolist()

        # Step 2: Compute embeddings only for sampled subset
        embeddings = compute_embeddings(texts, tokenizer, model)

        # Step 3: Evaluate clustering
        sil, purity, ari = evaluate_clustering(embeddings, labels)

        results.append({
            "data_type": "reference",
            "vector_type": "embedding_finetuned",
            "silhouette": sil,
            "purity": purity,
            "ARI_character": ari,
            "ARI_case": None
        })

    # ---------------------------
    # Simulation Data
    # ---------------------------

    print("Computing embeddings for simulation data...")

    sim_texts, sim_speakers, sim_cases = load_simulation_data()
    sim_embeddings = compute_embeddings(sim_texts, tokenizer, model)

    # 聚类
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sim_embeddings)

    kmeans = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=50,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)

    # character-level evaluation
    sil = silhouette_score(X_scaled, clusters)
    purity = compute_purity(sim_speakers, clusters)
    ari_character = adjusted_rand_score(sim_speakers, clusters)

    # case-level ARI
    ari_case = adjusted_rand_score(sim_cases, clusters)

    results.append({
        "data_type": "simulation",
        "vector_type": "embedding_finetuned",
        "silhouette": sil,
        "purity": purity,
        "ARI_character": ari_character,
        "ARI_case": ari_case
    })


# ------------------------------------------------------------------------------
# METRIC-BASED CLUSTERING
# ------------------------------------------------------------------------------

elif VECTOR_TYPE == "metric":

    print("Using metric-based vectors...")

    # Reference
    vectors_ref, labels_ref = load_metric_vectors("reference")
    sil, purity, ari = evaluate_clustering(vectors_ref, labels_ref)

    results.append({
        "data_type": "reference",
        "vector_type": "metric",
        "silhouette": sil,
        "purity": purity,
        "ARI_character": ari,
        "ARI_case": None
    })

    # Simulation
    vectors_sim, labels_sim = load_metric_vectors("simulation")
    sil, purity, ari = evaluate_clustering(vectors_sim, labels_sim)

    results.append({
        "data_type": "simulation",
        "vector_type": "metric",
        "silhouette": sil,
        "purity": purity,
        "ARI_character": ari,
        "ARI_case": None
    })


# ==============================================================================
# Save Results
# ==============================================================================

df_results = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "7.1_clustering_results.csv"
)

df_results.to_csv(
    output_file,
    index=False,
    float_format="%.4f"
)

print("\nClustering results saved to:", output_file)
print(df_results)