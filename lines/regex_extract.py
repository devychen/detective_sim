# Combine extract + clean
# 功能：提取指定角色的对话，并进行清理

import re
import csv
import os
import pandas as pd

def extract_quotes(input_dir, keywords, speak_verbs, min_tokens=10, max_tokens=100, context_window=80):
    """
    从指定目录的文本文件中提取特定角色的对话
    """
    lines = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

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

                candidate = candidate.replace('"', '').replace("“", "").replace("”", "").replace("‘", "").replace("’", "").replace("'", "")
                tokens = candidate.split()
                if min_tokens <= len(tokens) <= max_tokens:
                    lines.append(candidate)
    
    return lines

def clean_quotes(df, keywords):
    """
    清理台词，去掉包含角色名字、chapter 或括号的句子
    """
    def clean_quote(quote):
        if pd.isna(quote):
            return quote
        sentences = re.split(r'(?<=[.!?])\s+', quote)
        cleaned_sentences = []
        for sentence in sentences:
            if re.search(r'\b(' + '|'.join(keywords) + r'|chapter)\b', sentence, re.IGNORECASE):
                continue
            if "(" in sentence or ")" in sentence:
                continue
            cleaned_sentences.append(sentence)
        return ' '.join(cleaned_sentences).strip()
    
    df['quote'] = df['quote'].apply(clean_quote)
    return df

def main():
    # 角色配置
    character_configs = {
        'holmes': {
            'input_dir': '_novels/holmes',
            'keywords': ["holmes", "mr. holmes", "mr holmes"],
            'output_file': 'lines/cleaned_holmes_lines.csv'
        },
        'marple': {
            'input_dir': '_novels/marple',
            'keywords': ["marple", "ms. marple", "ms marple"],
            'output_file': 'lines/cleaned_marple_lines.csv'
        },
        'poirot': {
            'input_dir': '_novels/poirot',
            'keywords': ["poirot", "mr. poirot", "mr poirot"],
            'output_file': 'lines/cleaned_poirot_lines.csv'
        },
        'hastings': {
            'input_dir': '_novels/poirot',
            'keywords': ["hastings", "arthur hastings", "captain"],
            'output_file': 'lines/cleaned_hastings_lines.csv'
        },
        'watson': {
            'input_dir': '_novels/holmes',
            'keywords': ["watson", "john watson", "dr. watson"],
            'output_file': 'lines/cleaned_watson_lines.csv'
        },
        'japp': {
            'input_dir': '_novels/poirot',
            'keywords': ["japp", "james japp", "inspector japp", "inspector"],
            'output_file': 'lines/cleaned_japp_lines.csv'
        }
    }
    
    speak_verbs = [
        "says", "said", "replies", "replied", "asks", "asked", "cries", "cried",
        "answers", "answered", "remarks", "remarked", "observes", "observed", 
        "shouts", "shouted", "comments", "commented", "suggest", "suggested"
    ]
    
    os.makedirs("lines", exist_ok=True)
    
    for character, config in character_configs.items():
        print(f"\nProcessing {character}...")

        # 提取
        lines = extract_quotes(config['input_dir'], config['keywords'], speak_verbs)
        df = pd.DataFrame({"number": range(1, len(lines)+1), "quote": lines})

        # 清理
        df = clean_quotes(df, config['keywords'])

        # 保存最终 CSV
        df.to_csv(config['output_file'], index=False)
        print(f"Completed {character}, {len(df)} lines saved to {config['output_file']}")

if __name__ == "__main__":
    main()
