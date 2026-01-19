# train_lines.py
# combine three character's lines csv into one.
# Column: no., character, quote. 
# Balanced on total tokens.

import os
import pandas as pd
import random
import tiktoken  # pip install tiktoken

# === 参数 ===
lines_dir = "lines"
output_file = "lines/train_lines.csv"
files = {
    "holmes": "cleaned_holmes_lines.csv",
    "marple": "cleaned_marple_lines.csv",
    "poirot": "cleaned_poirot_lines.csv",
    "watson": "cleaned_watson_lines.csv",
    "japp": "cleaned_japp_lines.csv",
    "hastings": "cleaned_hastings_lines.csv"
}

# === 准备 tokenizer ===
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(str(text)))

# === 读取并统计 ===
data = {}
for character, filename in files.items():
    df = pd.read_csv(os.path.join(lines_dir, filename))
    df = df[["quote"]].copy()
    df.rename(columns={"quote": "quote"}, inplace=True)
    df["character"] = character
    df["tokens"] = df["quote"].apply(count_tokens)
    data[character] = df

# === 统计 token 总数 ===
totals = {ch: df["tokens"].sum() for ch, df in data.items()}
print("Original token counts：", totals)

# === 平衡逻辑 ===
watson_japp_hastings_total = sum(totals[ch] for ch in ["watson", "japp", "hastings"])
target = min(totals["holmes"], totals["marple"], totals["poirot"], watson_japp_hastings_total)
print("Targeted token counts (≈ 平衡值)：", target)

# 给三个主角取 target
balanced = {}
for ch in ["holmes", "marple", "poirot"]:
    df = data[ch].sample(frac=1, random_state=42)  # shuffle
    cumsum = df["tokens"].cumsum()
    cutoff_idx = cumsum.searchsorted(target + 500)
    # 防止超出范围
    cutoff_idx = min(cutoff_idx, len(df) - 1)
    balanced[ch] = df.iloc[:cutoff_idx + 1]

# Watson+Japp+Hastings 合起来取 target
combined = pd.concat([data["watson"], data["japp"], data["hastings"]]).sample(frac=1, random_state=42)
cumsum = combined["tokens"].cumsum()
cutoff_idx = cumsum.searchsorted(target + 100)
cutoff_idx = min(cutoff_idx, len(combined) - 1)
balanced["watson+japp+hastings"] = combined.iloc[:cutoff_idx + 1]

# === 合并导出 ===
final_df = pd.concat(balanced.values(), ignore_index=True)
final_df = final_df[["character", "quote"]].reset_index(drop=True)
final_df.insert(0, "no.", range(1, len(final_df) + 1))

final_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")

# === 打印最终 token 分布与行数 ===
print("\n=== Final token counts and line counts ===")
for ch in ["holmes", "marple", "poirot"]:
    df_ch = balanced[ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} lines")

for ch in ["watson", "japp", "hastings"]:
    df_ch = balanced["watson+japp+hastings"][balanced["watson+japp+hastings"]["character"] == ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} lines")

total_wjh = sum(balanced["watson+japp+hastings"]["tokens"])
total_rows_wjh = len(balanced["watson+japp+hastings"])
print(f"watson+japp+hastings in total: {total_wjh} tokens, {total_rows_wjh} lines")
