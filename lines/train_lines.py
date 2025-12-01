# train_lines.py
# combine three character's lines csv into one.
# Column: no., character, quote. 
# Balanced on total tokens, others kept full, final shuffle.

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

# === 平衡逻辑 for main characters ===
balanced = {}
for ch in ["holmes", "marple", "poirot"]:
    df = data[ch].sample(frac=1, random_state=42)  # shuffle
    # 取全部 token（原有设计就足够大）
    balanced[ch] = df

# === Others: Watson+Japp+Hastings 全量保留 ===
combined = pd.concat([data["watson"], data["japp"], data["hastings"]]).sample(frac=1, random_state=42)
balanced["others"] = combined

# === 合并所有数据 ===
final_df = pd.concat(balanced.values(), ignore_index=True)

# === 全局 shuffle ===
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === 添加行号 ===
final_df.insert(0, "no.", range(1, len(final_df) + 1))

# === 导出 CSV ===
final_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")

# === 打印最终 token 分布与行数 ===
print("\n=== Final token counts and line counts ===")
for ch in ["holmes", "marple", "poirot"]:
    df_ch = balanced[ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} lines")

others_df = balanced["others"]
for ch in ["watson", "japp", "hastings"]:
    df_ch = others_df[others_df["character"] == ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} lines")

total_others_tokens = others_df["tokens"].sum()
total_others_lines = len(others_df)
print(f"others in total: {total_others_tokens} tokens, {total_others_lines} lines")
