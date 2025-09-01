import pandas as pd
import re

# 读取 CSV 文件
df = pd.read_csv("lines/poirot_lines.csv")

# 定义处理函数
def clean_quote(quote):
    if pd.isna(quote):
        return quote
    # 1. 删除所有 quotation mark
    quote = quote.replace('"', '').replace("“", "").replace("”", "")
    
    # 2. 删除包含 'poirot' 或 'sherlock' 的整个句子
    # 使用正则按句号、问号、感叹号分割句子
    sentences = re.split(r'(?<=[.!?])\s+', quote)
    cleaned_sentences = []
    for sentence in sentences:
        # 如果句子里有 poirot 或 sherlock（忽略大小写），就跳过
        if re.search(r'\b(poirot|sherlock)\b', sentence, re.IGNORECASE):
            continue
        cleaned_sentences.append(sentence)
    
    # 合并剩下的句子
    return ' '.join(cleaned_sentences).strip()

# 应用处理函数
df['quote'] = df['quote'].apply(clean_quote)

# 保存为新的 CSV
df.to_csv("lines/cleaned_poirot_lines.csv", index=False)
