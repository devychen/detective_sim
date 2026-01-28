'''
Step 0  Path & config
Step 1  Load baseline & simulation data
Step 2  Sentence segmentation (only simulation)
Step 3  LLM annotation (batch + cache switch)
Step 4  Aggregate label distributions per character
Step 5  Compute KL divergence
Step 6  Save results

'''

import os
import glob
import json
import math
import time
import pandas as pd
import numpy as np
import requests
import re
from dotenv import load_dotenv
from collections import Counter, defaultdict
from tqdm import tqdm

# ----------------------------------------
# Step 0. Path settings & config
# ----------------------------------------

BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
MODEL_DIR = "./models/3class"
OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta/llama-3.2-3b-instruct"
load_dotenv("nvidia_key_3b.env")

API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Annotation switch (True = reuse existing annotation)
USE_EXISTING_ANNOTATION = True  

BATCH_SIZE = 8

LABELS = [
    "EVIDENCE_CITATION",
    "REASONING",
    "QUESTION",
    "HYPOTHESIS",
    "AGREEMENT",
    "DISAGREEMENT",
    "SOCIAL_TALK",
    "OTHER"
]

# ----------------------------------------
# Step 1. Load data
# ----------------------------------------

print("Loading baseline data...")
baseline_df = pd.read_csv(BASELINE_FILE)  # columns: quote, character

print("Loading simulation data...")
sim_files = glob.glob(DATA_GLOB)
sim_dfs = []
for f in sim_files:
    df = pd.read_csv(f)  # columns: turn, speaker, utterance, believed_murderer
    df["source_file"] = f
    sim_dfs.append(df)

sim_df = pd.concat(sim_dfs, ignore_index=True)

print(f"Baseline size: {len(baseline_df)}")
print(f"Simulation size: {len(sim_df)}")

# ----------------------------------------
# Step 2. Sentence segmentation (simulation only)
# ----------------------------------------

def split_sentences(text):
    # simple sentence splitter (you can replace with nltk/spacy)
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]

print("Splitting simulation utterances into sentences...")

sim_sentences = []
for _, row in sim_df.iterrows():
    sents = split_sentences(row["utterance"])
    for s in sents:
        sim_sentences.append({
            "turn": row["turn"],
            "speaker": row["speaker"],
            "sentence": s,
            "source_file": row["source_file"]
        })

sim_sent_df = pd.DataFrame(sim_sentences)

print(f"Simulation sentences: {len(sim_sent_df)}")

# ----------------------------------------
# Step 3. LLM annotation with batch + sentence-level cache
# ----------------------------------------

baseline_ann_file = os.path.join(OUTPUT_DIR, "5.1_baseline_annotations.csv")
sim_ann_file = os.path.join(OUTPUT_DIR, "5.1_simulation_annotations.csv")
cache_file = os.path.join(OUTPUT_DIR, "5.1_annotation_cache.json")

USE_EXISTING_ANNOTATION = True   # file-level switch
BATCH_SIZE = 20

LABELS = [
    "EVIDENCE_CITATION",
    "REASONING",
    "QUESTION",
    "HYPOTHESIS",
    "AGREEMENT",
    "DISAGREEMENT",
    "SOCIAL_TALK",
    "OTHER"
]

# -------- 3.1 Load cache --------

if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        ANNOTATION_CACHE = json.load(f)
    print(f"[Cache] Loaded {len(ANNOTATION_CACHE)} cached sentences.")
else:
    ANNOTATION_CACHE = {}
    print("[Cache] No cache found, starting fresh.")

# -------- 3.2 Prompt & API call --------

def build_prompt(batch_texts):
    label_desc = """
You are annotating dialogue acts in detective dialogues.
Choose exactly one label from the following list:

EVIDENCE_CITATION: citing clues or facts from the case
REASONING: logical inference or deduction
QUESTION: asking for information
HYPOTHESIS: proposing a possible explanation
AGREEMENT: expressing agreement
DISAGREEMENT: expressing disagreement or rebuttal
SOCIAL_TALK: casual or social conversation
OTHER: none of the above

Rules: 
1. Return ONLY a valid JSON array.
2. Do NOT include any explanation, markdown, or extra text.
3. The output length MUST equal the number of sentences.
4. Each element MUST be one of the labels above.
5. If uncertain, output "OTHER".

Sentences:

"""
    inputs = "\n".join([f"{i+1}. {t}" for i, t in enumerate(batch_texts)])
    return label_desc + "\nSentences:\n" + inputs

import re

def safe_parse_labels(content, expected_len):
    """
    Robust JSON extraction from LLM output.
    """
    try:
        return json.loads(content)
    except Exception:
        pass

    # try to extract JSON array using regex
    match = re.search(r"\[.*\]", content, re.S)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    print("⚠️ JSON parsing failed. Returning OTHER labels.")
    return ["OTHER"] * expected_len


def call_llm(batch_texts):
    prompt = build_prompt(batch_texts)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    
    labels = safe_parse_labels(content, len(batch_texts))


    if len(labels) != len(batch_texts):
        print("⚠️ Label count mismatch. Expected:", len(batch_texts), "Got:", len(labels))
        print("Raw output:", content)
        labels = ["OTHER"] * len(batch_texts)


    return labels

# -------- 3.3 Annotation function with cache --------

def annotate_texts(texts):
    global ANNOTATION_CACHE

    results = [None] * len(texts)
    uncached_texts = []
    uncached_indices = []

    # (1) check cache
    for idx, t in enumerate(texts):
        if t in ANNOTATION_CACHE:
            results[idx] = ANNOTATION_CACHE[t]
        else:
            uncached_texts.append(t)
            uncached_indices.append(idx)

    print(f"[Annotation] Total: {len(texts)}, Cached: {len(texts) - len(uncached_texts)}, To annotate: {len(uncached_texts)}")

    # (2) batch annotation for uncached texts
    new_labels = []
    for i in tqdm(range(0, len(uncached_texts), BATCH_SIZE)):
        batch = uncached_texts[i:i + BATCH_SIZE]
        labels = call_llm(batch)
        new_labels.extend(labels)
        time.sleep(0.3)  # avoid rate limit

    # (3) update cache
    for t, label in zip(uncached_texts, new_labels):
        ANNOTATION_CACHE[t] = label

    # (4) fill results
    for idx, label in zip(uncached_indices, new_labels):
        results[idx] = label

    # (5) save cache
    with open(cache_file, "w") as f:
        json.dump(ANNOTATION_CACHE, f, indent=2)

    print(f"[Cache] Updated cache size: {len(ANNOTATION_CACHE)}")

    return results

# -------- 3.4 Run annotation (baseline vs simulation) --------

if USE_EXISTING_ANNOTATION and os.path.exists(baseline_ann_file) and os.path.exists(sim_ann_file):
    print("[Annotation] Loading existing annotation files...")
    baseline_df = pd.read_csv(baseline_ann_file)
    sim_sent_df = pd.read_csv(sim_ann_file)
else:
    # Baseline: quote-level annotation
    print("[Annotation] Annotating baseline (quote-level)...")
    baseline_labels = annotate_texts(baseline_df["quote"].tolist())
    baseline_df["label"] = baseline_labels
    baseline_df.to_csv(baseline_ann_file, index=False)
    print(f"[Saved] Baseline annotations -> {baseline_ann_file}")

    # Simulation: sentence-level annotation
    print("[Annotation] Annotating simulation (sentence-level)...")
    sim_labels = annotate_texts(sim_sent_df["sentence"].tolist())
    sim_sent_df["label"] = sim_labels
    sim_sent_df.to_csv(sim_ann_file, index=False)
    print(f"[Saved] Simulation annotations -> {sim_ann_file}")


# ----------------------------------------
# Step 4. Compute label distributions per character
# ----------------------------------------

def compute_distribution(df, text_col, char_col):
    dist = {}
    for char in df[char_col].unique():
        labels = df[df[char_col] == char]["label"]
        counter = Counter(labels)
        total = sum(counter.values())
        probs = {l: counter.get(l, 0) / total for l in LABELS}
        dist[char] = probs
    return dist

baseline_dist = compute_distribution(baseline_df, "quote", "character")
sim_dist = compute_distribution(sim_sent_df, "sentence", "speaker")

# ----------------------------------------
# Step 5. KL divergence
# ----------------------------------------

def kl_divergence(p, q, eps=1e-9):
    kl = 0
    for label in LABELS:
        p_val = p.get(label, 0) + eps
        q_val = q.get(label, 0) + eps
        kl += p_val * math.log(p_val / q_val)
    return kl

kl_results = []

for char in baseline_dist:
    if char in sim_dist:
        kl = kl_divergence(sim_dist[char], baseline_dist[char])
        kl_results.append({
            "character": char,
            "kl_divergence": kl
        })

kl_df = pd.DataFrame(kl_results)

kl_file = os.path.join(OUTPUT_DIR, "5.1_discourse_kl_divergence.csv")
kl_df.to_csv(kl_file, index=False)

print(f"Saved KL divergence results to {kl_file}")

# ----------------------------------------
# Step 6. Save distributions (for descriptive analysis)
# ----------------------------------------

baseline_dist_file = os.path.join(OUTPUT_DIR, "5.1_baseline_label_distribution.json")
sim_dist_file = os.path.join(OUTPUT_DIR, "5.1_simulation_label_distribution.json")

with open(baseline_dist_file, "w") as f:
    json.dump(baseline_dist, f, indent=2)

with open(sim_dist_file, "w") as f:
    json.dump(sim_dist, f, indent=2)

print(f"Saved baseline label distribution to {baseline_dist_file}")
print(f"Saved simulation label distribution to {sim_dist_file}")
