import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# === 配置参数 ===
# =====================
lines_dir = "lines"
output_file = "merged_lines.csv"
files = {
    "holmes": "holmes_lines.csv",
    "marple": "marple_lines.csv",
    "poirot": "poirot_lines.csv",
    "watson": "waston_lines.csv",
    "japp": "japp_lines.csv",
    "hastings": "hastings_lines.csv"
}

# token 上限差值
TOKEN_MARGIN = 100
# 语义差异阈值（cosine similarity）
SIMILARITY_THRESHOLD = 0.8

# =====================
# === 初始化 LLM ===
# =====================
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("❌ NVIDIA_API_KEY 未设置，请在 .env 文件里添加")

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)
EMBEDDING_MODEL = "text-embedding-3-large"

# =====================
# === 读取 CSV 并准备数据 ===
# =====================
data = {}
for character, filename in files.items():
    df = pd.read_csv(os.path.join(lines_dir, filename))
    df = df[["quotes"]].copy()
    df.rename(columns={"quotes": "quote"}, inplace=True)
    df["character"] = character
    # token 计数（简单按空格粗略统计，可换成更精确的 tokenizer）
    df["tokens"] = df["quote"].apply(lambda x: len(str(x).split()))
    data[character] = df

# 统计原始 token 总数
totals = {ch: df["tokens"].sum() for ch, df in data.items()}
print("原始 token 数：", totals)

watson_japp_hastings_total = sum(totals[ch] for ch in ["watson", "japp", "hastings"])
target = min(totals["holmes"], totals["marple"], totals["poirot"], watson_japp_hastings_total)
print("目标 token 总数 (≈ 平衡值)：", target)

# =====================
# === LLM Embedding 函数 ===
# =====================
def get_embedding(text):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=str(text)
    )
    return np.array(resp.data[0].embedding)

# 为每句台词生成 embedding
for ch, df in data.items():
    print(f"正在生成 {ch} 的 embedding ...")
    df["embedding"] = df["quote"].apply(get_embedding)

# =====================
# === 贪心抽样函数 ===
# =====================
def select_lines(df, target_tokens):
    df = df.sample(frac=1, random_state=42)  # shuffle
    selected = []
    selected_tokens = 0

    for _, row in df.iterrows():
        if selected_tokens + row["tokens"] > target_tokens + TOKEN_MARGIN:
            continue
        # 检查与已选台词语义相似度
        if selected:
            sims = cosine_similarity(
                [row["embedding"]],
                [r["embedding"] for r in selected]
            )
            if np.any(sims >= SIMILARITY_THRESHOLD):
                continue
        selected.append(row)
        selected_tokens += row["tokens"]
    return pd.DataFrame(selected)

# =====================
# === 处理主角 ===
# =====================
balanced = {}
for ch in ["holmes", "marple", "poirot"]:
    balanced[ch] = select_lines(data[ch], target)

# =====================
# === 处理 Watson+Japp+Hastings ===
# =====================
combined = pd.concat([data["watson"], data["japp"], data["hastings"]])
balanced["watson+japp+hastings"] = select_lines(combined, target)

# =====================
# === 合并并导出 CSV ===
# =====================
final_df = pd.concat(balanced.values(), ignore_index=True)
final_df = final_df[["character", "quote"]].reset_index(drop=True)
final_df.insert(0, "no.", range(1, len(final_df) + 1))

final_df.to_csv(output_file, index=False)
print(f"\n已保存到 {output_file}")

# =====================
# === 打印最终 token 分布与行数 ===
# =====================
print("\n=== 最终 token 分布与行数 ===")
for ch in ["holmes", "marple", "poirot"]:
    df_ch = balanced[ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} 行")

for ch in ["watson", "japp", "hastings"]:
    df_ch = balanced["watson+japp+hastings"][balanced["watson+japp+hastings"]["character"] == ch]
    print(f"{ch}: {df_ch['tokens'].sum()} tokens, {len(df_ch)} 行")

total_wjh = sum(balanced["watson+japp+hastings"]["tokens"])
total_rows_wjh = len(balanced["watson+japp+hastings"])
print(f"watson+japp+hastings 总计: {total_wjh} tokens, {total_rows_wjh} 行")
