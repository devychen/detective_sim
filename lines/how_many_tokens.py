import csv
from transformers import AutoTokenizer

# 使用你指定的 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def count_tokens_in_csv(path):
    total_tokens = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 假设统计 "quote" 这一列
            text = row.get("quote", "")
            # return_tensors=False 保证我们得到的是纯 token id 列表
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)

    return total_tokens


if __name__ == "__main__":
    csv_path = "lines/holmes_lines.csv"  # 改成你的 CSV 文件路径
    total = count_tokens_in_csv(csv_path)
    print("Total tokens:", total)
