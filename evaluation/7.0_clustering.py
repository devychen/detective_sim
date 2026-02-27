# 7.1_clustering.py

"""
================================================================================
Clustering Analysis (Validation of Character Contamination in Simulation)
================================================================================

Purpose:
This script evaluates the separability of characters in the simulated dialogues
and in the reference corpus. The goal is to examine whether collaborative
simulation introduces "contamination" between character representations.

Design Overview:
1. Data Sources:
   - Reference corpus: baseline/train_lines_clean_balanced_3class.csv
       * Columns: quote, character
       * True speaker labels (ground truth)
   - Simulation data: data/*/*/dialogue_log.csv
       * Columns: turn, speaker, utterance, believed_murderer
       * 30 runs, 387 utterances total, balanced across characters
   - Metric vectors: from previous validation (optional)
       * Mandatory: lexical_similarity, character_distance, z_ref, depth, sentiment_distance
       * Optional: z_sim, intra_similarity

2. Vector Representations:
   - Approach 1: Embedding vectors
       * Use the same model as in cosine similarity / classifier embedding extraction
       * For BERT-based embeddings, take the [CLS] token hidden state
   - Approach 2: Metric vectors
       * Concatenate turn-wise metrics into a vector
       * Optional metrics controlled via switch

3. Clustering Procedure:
   - KMeans clustering (k=3, corresponding to 3 characters)
   - Perform separately on:
       * Reference corpus
       * Simulation data
   - Standardize features before clustering

4. Evaluation Metrics:
   - Silhouette Score: measures cluster cohesion and separation
   - Purity: proportion of majority true labels in each cluster
   - Adjusted Rand Index (ARI): corrected measure of agreement between clusters and true labels
   - These metrics quantify how well characters are separable and allow detection of contamination.

5. Outputs:
   - CSV: evaluation/7.1_clustering_results.csv
       * Columns: data_type, vector_type, silhouette, purity, ARI
================================================================================
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModel
import torch

# -------------------------------
# User settings
# -------------------------------
GOLD_PATH = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
METRIC_PATH = "evaluation/validation_turn_level_aggregated.csv"
OUTPUT_DIR = "evaluation"

VECTOR_TYPE = "embedding"  # "embedding" or "metric"
INCLUDE_OPTIONAL_METRICS = True  # switch for optional metrics in metric vector
N_BOOTSTRAP_REF = 5  # number of downsample bootstrap iterations for reference

K_CLUSTERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Functions
# -------------------------------

def load_reference_embeddings(tokenizer, model):
    """
    Step 1a: Load reference corpus and compute BERT embeddings
    """
    df_ref = pd.read_csv(GOLD_PATH)
    texts = df_ref["quote"].tolist()
    labels = df_ref["character"].tolist()
    
    embeddings = []
    batch_size = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            out = model(**enc)
            cls_emb = out.last_hidden_state[:,0,:].cpu().numpy()  # CLS token
            embeddings.append(cls_emb)
    embeddings = np.vstack(embeddings)
    return embeddings, labels, df_ref

def load_simulation_embeddings(tokenizer, model):
    """
    Step 1b: Load simulation dialogues and compute BERT embeddings
    """
    files = glob.glob(DATA_GLOB)
    all_texts = []
    all_labels = []
    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values(["turn"])
        all_texts.extend(df["utterance"].tolist())
        all_labels.extend(df["speaker"].tolist())
    # BERT embeddings
    embeddings = []
    batch_size = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            out = model(**enc)
            cls_emb = out.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(cls_emb)
    embeddings = np.vstack(embeddings)
    return embeddings, all_labels

def load_metric_vectors(data_type="simulation"):
    """
    Step 1c: Load concatenated metric vectors
    - data_type: "simulation" or "reference"
    """
    df_metrics = pd.read_csv(METRIC_PATH)
    mandatory = ["lexical_similarity","character_distance","z_ref","depth","sentiment_distance"]
    optional = ["z_sim","intra_similarity"]
    cols = mandatory + optional if INCLUDE_OPTIONAL_METRICS else mandatory
    # filter columns present in df
    cols = [c for c in cols if c in df_metrics.columns]
    
    if data_type=="simulation":
        # select only simulated utterances
        # assuming simulation data has run_id column
        df_sel = df_metrics.copy()
        df_sel = df_sel.dropna(subset=cols)
        vectors = df_sel[cols].values
        labels = df_sel["character"].tolist()
    else:
        # reference
        # merge metrics with reference quote? use only available metrics
        df_sel = df_metrics.copy()
        df_sel = df_sel.dropna(subset=cols)
        vectors = df_sel[cols].values
        labels = df_sel["character"].tolist()
    
    return vectors, labels

def evaluate_clustering(X, y_true):
    """
    Step 3: Fit KMeans, compute silhouette, purity, ARI
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=K_CLUSTERS, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Silhouette
    sil = silhouette_score(X_scaled, clusters)
    
    # Purity
    purity = np.mean([np.sum(np.array(y_true)[clusters==i] == 
                              max(set(np.array(y_true)[clusters==i]), key=list(np.array(y_true)[clusters==i]).count))
                      / np.sum(clusters==i) for i in range(K_CLUSTERS)])
    
    # ARI
    ari = adjusted_rand_score(y_true, clusters)
    
    return sil, purity, ari

# -------------------------------
# Main procedure
# -------------------------------

results = []

if VECTOR_TYPE=="embedding":
    print("Loading BERT model for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")  
    model = AutoModel.from_pretrained("google-bert/bert-base-cased").to(DEVICE)
    
    # Reference corpus embeddings
    print("Computing embeddings for reference corpus...")
    embeddings_ref, labels_ref, _ = load_reference_embeddings(tokenizer, model)
    # optional bootstrap downsample to match simulation size
    for i in range(N_BOOTSTRAP_REF):
        emb_sample, lbl_sample = resample(embeddings_ref, labels_ref, n_samples=387, random_state=42+i)
        sil, purity, ari = evaluate_clustering(emb_sample, lbl_sample)
        results.append({
            "data_type":"reference",
            "vector_type":"embedding",
            "silhouette":sil,
            "purity":purity,
            "ARI":ari
        })
    
    # Simulation embeddings
    print("Computing embeddings for simulation...")
    embeddings_sim, labels_sim = load_simulation_embeddings(tokenizer, model)
    sil, purity, ari = evaluate_clustering(embeddings_sim, labels_sim)
    results.append({
        "data_type":"simulation",
        "vector_type":"embedding",
        "silhouette":sil,
        "purity":purity,
        "ARI":ari
    })
    
elif VECTOR_TYPE=="metric":
    print("Loading metric vectors...")
    # reference
    vectors_ref, labels_ref = load_metric_vectors("reference")
    sil, purity, ari = evaluate_clustering(vectors_ref, labels_ref)
    results.append({
        "data_type":"reference",
        "vector_type":"metric",
        "silhouette":sil,
        "purity":purity,
        "ARI":ari
    })
    # simulation
    vectors_sim, labels_sim = load_metric_vectors("simulation")
    sil, purity, ari = evaluate_clustering(vectors_sim, labels_sim)
    results.append({
        "data_type":"simulation",
        "vector_type":"metric",
        "silhouette":sil,
        "purity":purity,
        "ARI":ari
    })

# -------------------------------
# Save results
# -------------------------------
df_results = pd.DataFrame(results)
output_file = os.path.join(OUTPUT_DIR, "7.1_clustering_results.csv")
df_results.to_csv(output_file, index=False, float_format="%.3f")
print(f"Clustering results saved to {output_file}")
print(df_results)