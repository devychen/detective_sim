# train.py
# train baseline + predict mode

import argparse
import pandas as pd
import random
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import torch
import sys


# =========================
# 0. NumPy 版本检测
# =========================
if int(np.__version__.split(".")[0]) >= 2:
    print("⚠️ 检测到 NumPy 版本 >= 2，可能和 PyTorch/Transformers 不兼容。")
    print("👉 建议执行: pip install 'numpy<2' --upgrade")
    sys.exit(1)


# =========================
# 1. 数据预处理
# =========================

def map_character(c):
    if c == "holmes":
        return "holmes"
    elif c == "poirot":
        return "poirot"
    elif c == "marple":
        return "marple"
    else:
        return "others"

def sample_by_tokens(df, tokenizer, max_tokens=5000):
    rows = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    total_tokens = 0
    sampled = []
    for _, row in rows.iterrows():
        quote = row["quote"]
        if pd.isna(quote) or not str(quote).strip():
            print(f"⚠️ 跳过无效台词: {row}")
            continue
        try:
            n_tokens = len(tokenizer.tokenize(str(quote)))
        except Exception as e:
            print(f"⚠️ Tokenizer 出错，跳过句子: {quote[:50]}... 错误: {e}")
            continue

        if total_tokens + n_tokens > max_tokens and len(sampled) > 0:
            break
        sampled.append(row)
        total_tokens += n_tokens
    print(f"✅ 采样类别 {df.iloc[0]['label']} 共 {len(sampled)} 句, ~{total_tokens} tokens")
    return pd.DataFrame(sampled)

def prepare_datasets(csv_path, tokenizer, max_tokens=5000):
    df = pd.read_csv(csv_path)
    df["label"] = df["character"].apply(map_character)

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    sampled_dfs = []
    for label in labels:
        part = df[df["label"] == label]
        sampled = sample_by_tokens(part, tokenizer, max_tokens=max_tokens)
        sampled_dfs.append(sampled)
    df_balanced = pd.concat(sampled_dfs).reset_index(drop=True)

    train_df, eval_df = train_test_split(
        df_balanced,
        test_size=0.1,
        stratify=df_balanced["label_id"],
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # ✅ Hugging Face Trainer 需要 `labels` 字段
    train_dataset = train_dataset.rename_column("label_id", "labels")
    eval_dataset = eval_dataset.rename_column("label_id", "labels")

    return train_dataset, eval_dataset, label2id, id2label


# =========================
# 2. Tokenizer & 数据处理
# =========================

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# 3. 主逻辑
# =========================

def train_model():
    model_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset, label2id, id2label = prepare_datasets(
        "lines/train_lines.csv", tokenizer, max_tokens=5000
    )

    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_dataset = train_dataset.remove_columns(["no.", "character", "quote", "label"])
    eval_dataset = eval_dataset.remove_columns(["no.", "character", "quote", "label"])

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./bert-classifier",
        eval_strategy="epoch",   # 你的环境里是 eval_strategy
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model("./bert-classifier")
    tokenizer.save_pretrained("./bert-classifier")
    print("\n✅ 模型训练完成并保存到 ./bert-classifier\n")


def predict_mode():
    model_dir = "./bert-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    print("\n✅ 已加载训练好的模型，可以开始测试 (输入 exit 退出)\n")
    while True:
        text = input("台词: ")
        if text.strip().lower() == "exit":
            break
        pred = classifier(text, truncation=True, max_length=128)
        # 只显示简洁结果
        label = pred[0]["label"]
        score = pred[0]["score"]
        print(f"预测角色: {label} (置信度 {score:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="predict",
                        help="选择运行模式: train 或 predict")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    else:
        predict_mode()


# python train.py --mode train
# python train.py --mode predict
