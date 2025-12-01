# llama_extract_lines.py

import os
import csv
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === 1. API key ===
load_dotenv("nvidia_key_3b.env")  # 设置你的环境变量
NVIDIA_KEY = os.getenv("NVIDIA_KEY_3B")  # 你的 key

# === 2. 模型初始化 ===
model_name = "meta/llama-3.2-3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.0,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=llm_pipe)

# === 3. prompt 模板 ===
PROMPT_TEMPLATE = """
你是小说分析助手。请从下面文本中提取角色台词，只提取指定角色：
Holmes, Poirot, Marple, Watson, Hastings, Japp

输出格式：CSV，每行一个台词：
character,quote

要求：
- 保留原文，不要改写
- 不要生成不存在的台词
- 不要输出无关文本

文本:
{text}
"""

# === 4. 函数：处理单个文本文件 ===
def extract_quotes_from_text(text, chunk_size=1500):
    """
    按 chunk_size token 分片调用 LLaMA
    返回 list of dict: [{'character': ..., 'quote': ...}, ...]
    """
    # 简单按行分片，也可改成 tokenizer token
    lines = text.split("\n")
    quotes = []

    chunk = []
    for line in lines:
        chunk.append(line)
        if len(" ".join(chunk).split()) >= chunk_size:
            prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
            output = llm(prompt)
            quotes.extend(parse_llm_output(output))
            chunk = []

    # 剩余 chunk
    if chunk:
        prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
        output = llm(prompt)
        quotes.extend(parse_llm_output(output))

    return quotes

# === 5. 解析 LLaMA 输出 ===
def parse_llm_output(text):
    """
    将 LLaMA 输出转换为 [{'character':..., 'quote':...}]
    假设输出每行 CSV: character,quote
    """
    result = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        if "," not in line:
            continue
        character, quote = line.split(",", 1)
        result.append({"character": character.strip(), "quote": quote.strip()})
    return result

# === 6. 批量处理目录 ===
def process_directory(input_dir, output_dir="lines"):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".txt"):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"Processing {filename} ...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        quotes = extract_quotes_from_text(text)

        # 按角色保存
        for character in ["holmes", "marple", "poirot", "watson", "hastings", "japp"]:
            char_quotes = [q["quote"] for q in quotes if q["character"].lower() == character]
            out_file = os.path.join(output_dir, f"cleaned_{character}_lines.csv")
            with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["number", "quote"])
                for i, quote in enumerate(char_quotes, 1):
                    writer.writerow([i, quote])
            print(f"Saved {len(char_quotes)} lines for {character} -> {out_file}")

if __name__ == "__main__":
    input_dir = "_novels"  # 小说目录
    process_directory(input_dir)
