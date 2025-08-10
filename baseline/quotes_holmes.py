import re
import csv

# 1. 读取原文
with open("_novels/the_return_of_sh.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. 匹配所有引号中的对话（支持英文和中英文引号）
dialogue_pattern = r'["“](.*?)["”]'
matches = list(re.finditer(dialogue_pattern, text, re.DOTALL))

sherlock_lines = []

# 3. 遍历每一句对话，判断是否是 Sherlock 说的
for match in matches:
    quote = match.group(1).strip()
    start, end = match.span()

    # 上下文范围（可调）
    context_before = text[max(0, start-80):start].lower()
    context_after = text[end:end+80].lower()

    # 关键词：包含不同的说法和动词
    sherlock_keywords = [
        "sherlock", "holmes", 
        "mr. holmes", "mr holmes"
    ]
    speak_verbs = [
        "said", "replied", "asked", "cried", "answered", "remarked", "observed", "shouted"
    ]

    # 判断条件
    if any(name in context_before for name in sherlock_keywords) and \
       any(verb in context_before for verb in speak_verbs):
        sherlock_lines.append(quote)
    elif any(name in context_after for name in sherlock_keywords) and \
         any(verb in context_after for verb in speak_verbs):
        sherlock_lines.append(quote)

# 4. 保存为 CSV
with open("quotes/the_return_of_sh.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["number", "quote"])
    for i, line in enumerate(sherlock_lines, 1):
        writer.writerow([i, line])

print(f"Extraction completed，in total found {len(sherlock_lines)} lines，Saved as the_return_of_sh.csv")
