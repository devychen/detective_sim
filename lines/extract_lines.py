# extract_quotes.py
# 通用脚本：提取不同角色的对话

import re
import csv
import os
import argparse

def extract_quotes(input_dir, keywords, speak_verbs, min_tokens=10, max_tokens=100, context_window=80):
    """
    从指定目录的文本文件中提取特定角色的对话
    
    Args:
        input_dir: 输入目录路径
        keywords: 角色关键词列表
        speak_verbs: 说话动词列表
        min_tokens: 最小token数量
        max_tokens: 最大token数量
        context_window: 上下文窗口大小
    
    Returns:
        提取的对话列表
    """
    lines = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 匹配所有引号中的内容
            dialogue_pattern = r'[""“](.*?)[""”]'
            matches = list(re.finditer(dialogue_pattern, text, re.DOTALL))

            for match in matches:
                quote = match.group(1).strip()
                start, end = match.span()

                context_before = text[max(0, start-context_window):start].lower()
                context_after = text[end:end+context_window].lower()

                if (any(name in context_before for name in keywords) and
                    any(verb in context_before for verb in speak_verbs)):
                    candidate = quote
                elif (any(name in context_after for name in keywords) and
                      any(verb in context_after for verb in speak_verbs)):
                    candidate = quote
                else:
                    continue

                # 清理引号
                candidate = candidate.replace('"', '').replace("“", "").replace("”", "").replace("‘", "").replace("’", "").replace("'", "")

                # 限制长度
                tokens = candidate.split()
                if min_tokens <= len(tokens) <= max_tokens:
                    lines.append(candidate)
    
    return lines

def main():
    # 配置参数
    parser = argparse.ArgumentParser(description='Extract character quotes from novels')
    parser.add_argument('character', choices=['holmes', 'marple', 'poirot', 'hastings', 'watson', 'japp'], 
                       help='Character to extract quotes for')
    args = parser.parse_args()
    
    # 角色配置（现在所有角色的min_tokens都是10）
    character_configs = {
        'holmes': {
            'input_dir': '_novels/holmes',
            'keywords': ["holmes", "holmes", "holmes holmes", "mr. holmes", "mr holmes"],
            'output_file': 'lines/holmes_lines.csv'
        },
        'marple': {
            'input_dir': '_novels/marple',
            'keywords': ["marple", "marple", "marple marple", "ms. marple", "ms marple"],
            'output_file': 'lines/marple_lines.csv'
        },
        'poirot': {
            'input_dir': '_novels/poirot',
            'keywords': ["poirot", "poirot", "poirot poirot", "mr. poirot", "mr poirot"],
            'output_file': 'lines/poirot_lines.csv'
        },
        'hastings': {
            'input_dir': '_novels/poirot',
            'keywords': ["hastings", "arthur hastings", "mr. hastings", "mr hastings", 
                        "captain arthur hastings", "captain"],
            'output_file': 'lines/hastings_lines.csv'
        },
        'watson': {
            'input_dir': '_novels/holmes',
            'keywords': ["watson", "john watson", "john h. watson", "mr. watson", 
                        "mr watson", "dr. watson"],
            'output_file': 'lines/watson_lines.csv'
        },
        'japp': {
            'input_dir': '_novels/poirot',
            'keywords': ["japp", "james japp", "mr. japp", "mr japp", "inspector japp", "inspector"],
            'output_file': 'lines/japp_lines.csv'
        }
    }
    
    # 通用的说话动词列表
    speak_verbs = [
        "says", "said", "replies", "replied", "asks", "asked", "cries", "cried",
        "answers", "answered", "remarks", "remarked", "observes", "observed", 
        "shouts", "shouted", "comments", "commented", "suggest", "suggested"
    ]
    
    # 获取配置
    config = character_configs[args.character]
    
    # 提取对话（所有角色都使用min_tokens=10）
    lines = extract_quotes(
        config['input_dir'], 
        config['keywords'], 
        speak_verbs,
        min_tokens=10  # 统一设置为10
    )
    
    # 保存到CSV
    os.makedirs("lines", exist_ok=True)
    with open(config['output_file'], "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["number", "quote"])
        for i, line in enumerate(lines, 1):
            writer.writerow([i, line])
    
    print(f"Extraction completed for {args.character}, found {len(lines)} lines in total")

if __name__ == "__main__":
    main()