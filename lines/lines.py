import pandas as pd
import random
import re

# 配置文件路径
files = {
    "sh": "lines/sherlock_lines.csv",
    "hp": "lines/poirot_lines.csv",
    "mm": "lines/marple_lines.csv"
}

def clean_text(text):
    # 去掉标点符号
    text_no_punct = re.sub(r"[^\w\s]", "", text)
    return text_no_punct

def filter_by_token_count(quotes, min_tokens=8):
    filtered = []
    for q in quotes:
        q_clean = clean_text(q)
        tokens = q_clean.split()
        if len(tokens) >= min_tokens:
            filtered.append(q)
    return filtered

final_data = []
for char, filepath in files.items():
    # 读取 CSV
    df = pd.read_csv(filepath)
    
    # 取出台词列
    if "quote" in df.columns:
        quotes = df["quote"].dropna().tolist()
    else:
        raise ValueError(f"{filepath} 没有 'quote' 列")

    # 过滤短句
    filtered_quotes = filter_by_token_count(quotes, min_tokens=8)

    # 随机抽取 500 条
    if len(filtered_quotes) > 500:
        sampled_quotes = random.sample(filtered_quotes, 500)
    else:
        sampled_quotes = filtered_quotes

    # 加入总数据
    for q in sampled_quotes:
        final_data.append([char, q])

# 打乱顺序（可选）
random.shuffle(final_data)

# 写入总 CSV
output_rows = []
for i, (char, quote) in enumerate(final_data, 1):
    output_rows.append([i, char, quote])

df_out = pd.DataFrame(output_rows, columns=["number", "character", "quote"])
df_out.to_csv("combined_quotes.csv", index=False, encoding="utf-8")

print(f"合并完成，总共 {len(final_data)} 条台词，保存到 combined_lines.csv")
