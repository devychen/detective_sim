import pandas as pd
import re

def clean_file(input_path, output_path, keywords):
    # 读取 CSV
    df = pd.read_csv(input_path)

    def clean_quote(quote):
        if pd.isna(quote):
            return quote
        # 删除引号
        quote = quote.replace('"', '').replace("“", "").replace("”", "")
        # 按句子拆分
        sentences = re.split(r'(?<=[.!?])\s+', quote)
        cleaned_sentences = []
        for sentence in sentences:
            if re.search(r'\b(' + '|'.join(keywords) + r'|chapter)\b',
                         sentence, re.IGNORECASE):
                continue
            cleaned_sentences.append(sentence)
        return ' '.join(cleaned_sentences).strip()

    # 应用清理函数
    df['quote'] = df['quote'].apply(clean_quote)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # 配置三个人物的文件
    configs = [
        ("lines/holmes_lines.csv", "lines/cleaned_holmes_lines.csv", ["holmes", "sherlock", "chapter"]),
        ("lines/marple_lines.csv", "lines/cleaned_marple_lines.csv", ["marple", "chapter", "chapter"]),
        ("lines/poirot_lines.csv", "lines/cleaned_poirot_lines.csv", ["poirot", "hercule" , "chapter"]),
        ("lines/hastings_lines.csv", "lines/cleaned_hastings_lines.csv", ["hastings", "arthur", "captain", "chapter"]),
        ("lines/watson_lines.csv", "lines/cleaned_watson_lines.csv", ["watson", "john", "chapter"]),
        ("lines/japp_lines.csv", "lines/cleaned_japp_lines.csv", ["japp", "inspecter", "chapter"]),

    ]

    for input_file, output_file, keys in configs:
        clean_file(input_file, output_file, keys)
