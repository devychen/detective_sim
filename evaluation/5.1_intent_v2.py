'''
Step 1: Load baseline & simulation data
Step 2: Sentence splitting (simulation only)
Step 3: LLM annotation (batch + cache)
Step 4: Aggregate intent distributions per character
Step 5: Compute KL divergence (baseline vs simulation)
Step 6: Save results

'''

import os
import glob
import json
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import requests
from collections import Counter
import spacy

# ---------------------------------------- #
# Step 0. Path settings
# ---------------------------------------- #

BASELINE_FILE = "baseline/train_lines_clean_balanced_3class.csv"
DATA_GLOB = "data/*/*/dialogue_log.csv"
MODEL_DIR = "./models/3class"
OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------- #
# Step 0.1 LLM settings
# ---------------------------------------- #

MODEL_NAME = "meta/llama-3.2-3b-instruct"
load_dotenv("nvidia_key_3b.env")

API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# annotation switch (avoid re-run)
RUN_ANNOTATION = True   # set False if you want to skip annotation and reuse cached results

BATCH_SIZE = 5
SLEEP_TIME = 1.5

# ---------------------------------------- #
# Step 0.2 Label set
# ---------------------------------------- #

LABELS = [
    "EVIDENCE",
    "REASONING",
    "QUESTION",
    "HYPOTHESIS",
    "AGREEMENT",
    "DISAGREEMENT",
    "SOCIAL",
    "OTHER"
]

LABEL2ID = {l:i for i,l in enumerate(LABELS)}

# ---------------------------------------- #
# Step 1. Load data
# ---------------------------------------- #

def load_baseline():
    df = pd.read_csv(BASELINE_FILE)
    # columns: quote, character
    return df

def load_simulation():
    files = glob.glob(DATA_GLOB)
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

baseline_df = load_baseline()
sim_df = load_simulation()

print(f"[INFO] Loaded baseline: {len(baseline_df)} lines")
print(f"[INFO] Loaded simulation: {len(sim_df)} utterances")

# ---------------------------------------- #
# Step 2. Sentence splitting (simulation only)
# ---------------------------------------- #

import spacy

# load once globally (avoid repeated loading)
nlp = spacy.load("en_core_web_sm")

def split_sentences(df):
    rows = []
    for _, r in df.iterrows():
        speaker = r["speaker"]
        utterance = str(r["utterance"])
        
        doc = nlp(utterance)
        sentences = [sent.text for sent in doc.sents]
        
        for s in sentences:
            s = s.strip()
            if s:
                rows.append({
                    "speaker": speaker,
                    "sentence": s,
                    "source_file": r["source_file"]
                })
    return pd.DataFrame(rows)

sim_sent_df = split_sentences(sim_df)
print(f"[INFO] Simulation sentences: {len(sim_sent_df)}")


# ---------------------------------------- #
# Step 3. Annotation cache
# ---------------------------------------- #

CACHE_FILE = os.path.join(OUTPUT_DIR, "5.1_annotation_cache.json")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

print(f"[INFO] Cache size: {len(CACHE)}")

# ---------------------------------------- #
# Step 4. LLM annotation function
# ---------------------------------------- #

def build_prompt(sentences):
    label_str = ", ".join(LABELS)
    examples = "\n".join([f"{i+1}. {s}" for i,s in enumerate(sentences)])
    
    prompt = f"""
Classify each sentence into one label from: {label_str}.
Return only labels in order, one per line.

Sentences:
{examples}
"""
    return prompt.strip()

def call_llm(sentences):
    prompt = build_prompt(sentences)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print(resp.text)
        raise RuntimeError("API error")
    
    text = resp.json()["choices"][0]["message"]["content"]
    labels = [l.strip() for l in text.split("\n") if l.strip()]
    return labels

def annotate_texts(texts):
    results = []
    batch = []
    batch_idx = []

    for i, t in enumerate(texts):
        if t in CACHE:
            results.append(CACHE[t])
        else:
            batch.append(t)
            batch_idx.append(i)
        
        if len(batch) == BATCH_SIZE or (i == len(texts)-1 and batch):
            labels = call_llm(batch)
            if len(labels) != len(batch):
                labels = ["OTHER"] * len(batch)
            
            for t_, l_ in zip(batch, labels):
                if l_ not in LABELS:
                    l_ = "OTHER"
                CACHE[t_] = l_
                results.append(l_)
            
            batch = []
            batch_idx = []
            time.sleep(SLEEP_TIME)
    
    return results

# ---------------------------------------- #
# Step 5. Run annotation
# ---------------------------------------- #

if RUN_ANNOTATION:
    
    print("[INFO] Annotating baseline (quote-level)...")
    baseline_texts = baseline_df["quote"].tolist()
    baseline_labels = annotate_texts(baseline_texts)
    baseline_df["intent"] = baseline_labels
    
    print("[INFO] Annotating simulation (sentence-level)...")
    sim_texts = sim_sent_df["sentence"].tolist()
    sim_labels = annotate_texts(sim_texts)
    sim_sent_df["intent"] = sim_labels
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False, indent=2)
    
    baseline_out = os.path.join(OUTPUT_DIR, "5.1_baseline_annotated.csv")
    sim_out = os.path.join(OUTPUT_DIR, "5.1_simulation_annotated.csv")
    
    baseline_df.to_csv(baseline_out, index=False)
    sim_sent_df.to_csv(sim_out, index=False)
    
    print(f"[INFO] Baseline annotation saved to: {baseline_out}")
    print(f"[INFO] Simulation annotation saved to: {sim_out}")

else:
    baseline_df = pd.read_csv(os.path.join(OUTPUT_DIR, "5.1_baseline_annotated.csv"))
    sim_sent_df = pd.read_csv(os.path.join(OUTPUT_DIR, "5.1_simulation_annotated.csv"))
    print("[INFO] Loaded annotated data from disk")

# ---------------------------------------- #
# Step 6. Compute intent distributions
# ---------------------------------------- #

def compute_distribution(df, speaker_col, intent_col):
    result = {}
    for speaker, g in df.groupby(speaker_col):
        counts = Counter(g[intent_col])
        total = sum(counts.values())
        dist = np.array([counts.get(l, 0)/total for l in LABELS])
        result[speaker] = dist
    return result

baseline_dist = compute_distribution(baseline_df, "character", "intent")
sim_dist = compute_distribution(sim_sent_df, "speaker", "intent")

# ---------------------------------------- #
# Step 7. KL divergence
# ---------------------------------------- #

def kl_divergence(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

rows = []

for character in baseline_dist:
    if character in sim_dist:
        kl = kl_divergence(sim_dist[character], baseline_dist[character])
        rows.append({
            "character": character,
            "kl_divergence": kl
        })

kl_df = pd.DataFrame(rows)
kl_out = os.path.join(OUTPUT_DIR, "5.1_kl_divergence.csv")
kl_df.to_csv(kl_out, index=False)

print(f"[INFO] KL divergence results saved to: {kl_out}")

# ---------------------------------------- #
# Step 8. Save distributions (for analysis)
# ---------------------------------------- #

dist_rows = []

for c, d in baseline_dist.items():
    for label, val in zip(LABELS, d):
        dist_rows.append({
            "character": c,
            "source": "baseline",
            "label": label,
            "proportion": val
        })

for c, d in sim_dist.items():
    for label, val in zip(LABELS, d):
        dist_rows.append({
            "character": c,
            "source": "simulation",
            "label": label,
            "proportion": val
        })

dist_df = pd.DataFrame(dist_rows)
dist_out = os.path.join(OUTPUT_DIR, "5.1_intent_distribution.csv")
dist_df.to_csv(dist_out, index=False)

print(f"[INFO] Intent distributions saved to: {dist_out}")
