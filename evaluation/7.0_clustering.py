# ----------------------------------------
# Step 0. Imports and Path Settings
# ----------------------------------------
import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sentence_transformers import SentenceTransformer  # 用于 embedding

BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
MODEL_DIR = "./models/3class"
OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding model
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # 可根据需要换成自己的模型

# ----------------------------------------
# Step 1. Load Data
# ----------------------------------------
# Reference corpus (baseline)
baseline_df = pd.read_csv(BASELINE_FILE)  # columns: quote, character

# Simulation dialogues
sim_files = glob.glob(DATA_GLOB)
sim_dfs = [pd.read_csv(f) for f in sim_files]
sim_df = pd.concat(sim_dfs, ignore_index=True)  # columns: turn, speaker, utterance, believed_murderer

# ----------------------------------------
# Step 2. Vectorize Utterances
# ----------------------------------------
# Approach 1: Use embedding model + optional PCA
def embed_and_reduce(texts, dim=50):
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True)
    if dim < embeddings.shape[1]:
        pca = PCA(n_components=dim)
        embeddings = pca.fit_transform(embeddings)
    return embeddings

# Reference corpus embeddings
baseline_vectors = embed_and_reduce(baseline_df['quote'].tolist())
# Simulation embeddings
sim_vectors = embed_and_reduce(sim_df['utterance'].tolist())

# Approach 2: Concatenate turn-wise metrics (示例用简单统计作为 placeholder)
# 可以改成你之前计算的 ooc 相关 metrics
def get_turn_metrics(df):
    # 这里用简单示例：utterance length, word count, avg word length
    metrics = []
    for u in df['utterance']:
        words = u.split()
        length = len(u)
        wc = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if wc > 0 else 0
        metrics.append([length, wc, avg_word_len])
    return np.array(metrics)

baseline_metrics = get_turn_metrics(baseline_df)
sim_metrics = get_turn_metrics(sim_df)

# ----------------------------------------
# Step 3. Clustering
# ----------------------------------------
def cluster_and_evaluate(vectors, labels_true, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # Silhouette Coefficient
    sil_score = silhouette_score(vectors, cluster_labels)
    
    # Purity
    def purity_score(y_true, y_pred):
        # group by cluster
        cluster_purity = []
        for c in np.unique(y_pred):
            idx = np.where(y_pred == c)[0]
            cluster_labels_count = Counter(y_true[idx])
            most_common = cluster_labels_count.most_common(1)[0][1]
            cluster_purity.append(most_common / len(idx))
        return np.mean(cluster_purity)
    
    pur = purity_score(np.array(labels_true), cluster_labels)
    
    return cluster_labels, sil_score, pur

# ----------------------------------------
# Step 4. Apply Clustering
# ----------------------------------------
# Reference corpus
baseline_labels = baseline_df['character'].tolist()
baseline_emb_cluster, baseline_emb_sil, baseline_emb_pur = cluster_and_evaluate(baseline_vectors, baseline_labels)
baseline_metric_cluster, baseline_metric_sil, baseline_metric_pur = cluster_and_evaluate(baseline_metrics, baseline_labels)

# Simulation
sim_labels = sim_df['speaker'].tolist()
sim_emb_cluster, sim_emb_sil, sim_emb_pur = cluster_and_evaluate(sim_vectors, sim_labels)
sim_metric_cluster, sim_metric_sil, sim_metric_pur = cluster_and_evaluate(sim_metrics, sim_labels)

# ----------------------------------------
# Step 5. Save Results
# ----------------------------------------
results = {
    'baseline_emb_sil': baseline_emb_sil,
    'baseline_emb_pur': baseline_emb_pur,
    'baseline_metric_sil': baseline_metric_sil,
    'baseline_metric_pur': baseline_metric_pur,
    'sim_emb_sil': sim_emb_sil,
    'sim_emb_pur': sim_emb_pur,
    'sim_metric_sil': sim_metric_sil,
    'sim_metric_pur': sim_metric_pur
}

results_df = pd.DataFrame([results])
results_df.to_csv(os.path.join(OUTPUT_DIR, "5.1_clustering_evaluation.csv"), index=False)
print("Clustering evaluation saved to", OUTPUT_DIR)
