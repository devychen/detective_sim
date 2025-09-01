# extract_watson.py
# Aim: extract other character's quotes from original books
# Use captain arthur Hastings


import re
import csv
import os

# 1. 目录路径
input_dir = "_novels/poirot"

# 2. 关键词配置
watson_keywords = [
    "hastings", 
    "arthur hastings", 
    "mr. hastings", 
    "mr hastings",
    "captain arthur hastings",
    "captain"
]
speak_verbs = [
    "says", "said", 
    "replies", "replied", 
    "asks", "asked", 
    "cries", "cried",
    "answers", "answered", 
    "remarks", "remarked", 
    "observes", "observed", 
    "shouts", "shouted",
    "comments", "commented",
    "suggest", "suggested"
]

# 3. 结果列表
watson_lines = []

# 4. 遍历目录下所有 txt 文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".txt"):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 匹配所有引号中的内容
        dialogue_pattern = r'["“](.*?)["”]'
        matches = list(re.finditer(dialogue_pattern, text, re.DOTALL))

        for match in matches:
            quote = match.group(1).strip()
            start, end = match.span()

            context_before = text[max(0, start-80):start].lower()
            context_after = text[end:end+80].lower()

            if (any(name in context_before for name in watson_keywords) and
                any(verb in context_before for verb in speak_verbs)):
                candidate = quote
            elif (any(name in context_after for name in watson_keywords) and
                  any(verb in context_after for verb in speak_verbs)):
                candidate = quote
            else:
                continue

            # ---- 新增约束 ----
            # 1. 去掉引号（包括单双引号和特殊样式引号）
            candidate = candidate.replace('"', '').replace("“", "").replace("”", "").replace("‘", "").replace("’", "").replace("'", "")

            # 2. 限制长度 5-100 tokens
            tokens = candidate.split()
            if 5 <= len(tokens) <= 100:
                watson_lines.append(candidate)

# 5. 保存到 CSV
os.makedirs("lines", exist_ok=True)
with open("lines/hastings_lines.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["number", "quote"])
    for i, line in enumerate(watson_lines, 1):
        writer.writerow([i, line])

print(f"Extraction completed, found {len(watson_lines)} lines in total")
