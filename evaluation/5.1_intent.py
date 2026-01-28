"""
Step 1: Load baseline & simulation data
Step 2: Sentence splitting (baseline + simulation)
Step 3: Baseline sampling by character
Step 4: LLM annotation (batch + cache)
Step 5: Aggregate intent distributions per character
Step 6: Compute KL divergence (baseline vs simulation)
Step 7: Save results


"""

import os
import glob
import json
import time
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
OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------- #
# Step 0.1 LLM settings
# ---------------------------------------- #

MODEL_NAME = "meta/llama-3.2-3b-instruct"
load_dotenv("nvidia_key_3b.env")

API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

RUN_ANNOTATION = True   # False = reuse cached annotation

SLEEP_TIME = 0.3  # 降低 sleep

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
# Step 2. Sentence splitting (baseline + simulation)
# ---------------------------------------- #

nlp = spacy.load("en_core_web_sm")

def split_sentences_baseline(df):
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Splitting baseline"):
        character = r["character"]
        text = str(r["quote"])
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                rows.append({
                    "character": character,
                    "sentence": s
                })
    return pd.DataFrame(rows)

def split_sentences_simulation(df):
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Splitting simulation"):
        speaker = r["speaker"]
        utterance = str(r["utterance"])
        doc = nlp(utterance)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                rows.append({
                    "speaker": speaker,
                    "sentence": s,
                    "source_file": r["source_file"]
                })
    return pd.DataFrame(rows)

baseline_sent_df = split_sentences_baseline(baseline_df)
sim_sent_df = split_sentences_simulation(sim_df)

print(f"[INFO] Baseline sentences: {len(baseline_sent_df)}")
print(f"[INFO] Simulation sentences: {len(sim_sent_df)}")

# ---------------------------------------- #
# Step 3. Baseline sampling by character
# ---------------------------------------- #

SAMPLE_PER_CHARACTER = 1000   

def sample_baseline_by_character(df, n_per_char):
    samples = []
    for c, g in df.groupby("character"):
        n = min(len(g), n_per_char)
        sampled = g.sample(n, random_state=42)
        samples.append(sampled)
        print(f"[INFO] Sampled {n} sentences for character: {c}")
    return pd.concat(samples, ignore_index=True)

baseline_sample_df = sample_baseline_by_character(baseline_sent_df, SAMPLE_PER_CHARACTER)

print(f"[INFO] Total baseline sampled sentences: {len(baseline_sample_df)}")

# ---------------------------------------- #
# Step 4. Annotation cache
# ---------------------------------------- #

CACHE_FILE = os.path.join(OUTPUT_DIR, "intent_annotation_cache.json")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

print(f"[INFO] Cache size: {len(CACHE)}")

# ---------------------------------------- #
# Step 5. LLM annotation
# ---------------------------------------- #

def build_prompt(sentences):
    label_str = ", ".join(LABELS)
    examples = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    prompt = f"""
Classify each sentence into one label from: {label_str}.
Return only labels in order, one per line.

Sentences:
{examples}
"""
    return prompt.strip()

def call_llm(sentences, retry=5):
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

    for i in range(retry):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            labels = [l.strip() for l in text.split("\n") if l.strip()]
            return labels
        except Exception as e:
            wait = 2 ** i
            print(f"[WARN] LLM call failed ({i+1}/{retry}), wait {wait}s: {e}")
            time.sleep(wait)

    return ["OTHER"] * len(sentences)


MAX_CHARS_PER_BATCH = 2000   # 核心参数（建议 1500~3000）
SLEEP_TIME = 0.5

def annotate_texts(texts):
    results = []
    batch = []
    batch_chars = 0

    for t in tqdm(texts, desc="Annotating"):
        if t in CACHE:
            results.append(CACHE[t])
            continue

        batch.append(t)
        batch_chars += len(t)

        # ⭐ 动态 batch 控制
        if batch_chars >= MAX_CHARS_PER_BATCH:
            labels = call_llm(batch)
            if len(labels) != len(batch):
                labels = ["OTHER"] * len(batch)

            for s, l in zip(batch, labels):
                if l not in LABELS:
                    l = "OTHER"
                CACHE[s] = l
                results.append(l)

            batch = []
            batch_chars = 0
            time.sleep(SLEEP_TIME)

    # last batch
    if batch:
        labels = call_llm(batch)
        if len(labels) != len(batch):
            labels = ["OTHER"] * len(batch)
        for s, l in zip(batch, labels):
            if l not in LABELS:
                l = "OTHER"
            CACHE[s] = l
            results.append(l)

    return results


# ---------------------------------------- #
# Step 6. Run annotation
# ---------------------------------------- #

if RUN_ANNOTATION:
    print("[INFO] Annotating baseline sample...")
    baseline_labels = annotate_texts(baseline_sample_df["sentence"].tolist())
    baseline_sample_df["intent"] = baseline_labels

    print("[INFO] Annotating simulation...")
    sim_labels = annotate_texts(sim_sent_df["sentence"].tolist())
    sim_sent_df["intent"] = sim_labels

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False, indent=2)

    baseline_sample_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_annotated.csv"), index=False)
    sim_sent_df.to_csv(os.path.join(OUTPUT_DIR, "simulation_annotated.csv"), index=False)

else:
    baseline_sample_df = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_annotated.csv"))
    sim_sent_df = pd.read_csv(os.path.join(OUTPUT_DIR, "simulation_annotated.csv"))

# ---------------------------------------- #
# Step 7. Compute intent distributions
# ---------------------------------------- #

def compute_distribution(df, speaker_col, intent_col):
    result = {}
    for speaker, g in df.groupby(speaker_col):
        counts = Counter(g[intent_col])
        total = sum(counts.values())
        dist = np.array([counts.get(l, 0) / total for l in LABELS])
        result[speaker] = dist
    return result

baseline_dist = compute_distribution(baseline_sample_df, "character", "intent")
sim_dist = compute_distribution(sim_sent_df, "speaker", "intent")

# ---------------------------------------- #
# Step 8. KL divergence (optional JS divergence)
# ---------------------------------------- #

def kl_divergence(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def js_divergence(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

rows = []

for character in baseline_dist:
    if character in sim_dist:
        kl = kl_divergence(sim_dist[character], baseline_dist[character])
        js = js_divergence(sim_dist[character], baseline_dist[character])
        rows.append({
            "character": character,
            "kl_divergence": kl,
            "js_divergence": js
        })

kl_df = pd.DataFrame(rows)
kl_out = os.path.join(OUTPUT_DIR, "intent_divergence.csv")
kl_df.to_csv(kl_out, index=False)

print(f"[INFO] Divergence results saved to: {kl_out}")

# ---------------------------------------- #
# Step 9. Save distributions
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
dist_out = os.path.join(OUTPUT_DIR, "intent_distribution.csv")
dist_df.to_csv(dist_out, index=False)

print(f"[INFO] Intent distributions saved to: {dist_out}")
