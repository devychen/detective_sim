# combine lines
import pandas as pd
import random
import re

files = {
    "sh": "lines/sherlock_lines.csv",
    "hp": "lines/poirot_lines.csv",
    "mm": "lines/marple_lines.csv"
}

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text)

def filter_by_token_count(quotes, min_tokens=8):
    return [q for q in quotes if len(clean_text(q).split()) >= min_tokens]

def sample_by_token_limit(quotes, target_tokens=10000, tolerance=500):
    random.shuffle(quotes)  # 随机打乱
    selected = []
    total_tokens = 0
    for q in quotes:
        tokens_in_q = len(clean_text(q).split())
        if total_tokens + tokens_in_q > target_tokens + tolerance:
            continue
        selected.append(q)
        total_tokens += tokens_in_q
        if total_tokens >= target_tokens - tolerance:
            break
    return selected, total_tokens

final_data = []
token_counts = {}

TARGET_TOKENS = 10000
TOLERANCE = 500

for char, filepath in files.items():
    df = pd.read_csv(filepath)
    if "quote" not in df.columns:
        raise ValueError(f"{filepath} 没有 'quote' 列")

    quotes = df["quote"].dropna().tolist()
    filtered_quotes = filter_by_token_count(quotes, min_tokens=8)

    sampled_quotes, total_tokens = sample_by_token_limit(
        filtered_quotes,
        target_tokens=TARGET_TOKENS,
        tolerance=TOLERANCE
    )

    token_counts[char] = total_tokens

    for q in sampled_quotes:
        final_data.append([char, q])

random.shuffle(final_data)

output_rows = []
for i, (char, quote) in enumerate(final_data, 1):
    output_rows.append([i, char, quote])

df_out = pd.DataFrame(output_rows, columns=["number", "character", "quote"])
df_out.to_csv("lines/train.csv", index=False, encoding="utf-8")

print(f"Combined in total {len(final_data)} lines，saved to train.csv.")

print("\nToken counts by character:")
for char, count in token_counts.items():
    print(f"{char}: {count} tokens")
