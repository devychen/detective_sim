# add accuracy
# split 80/10/10 train/val/test

# train.py — baseline model training with train/val/test split + accuracy metrics

import pandas as pd
import random
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
# 0. 调试信息
# =========================
import transformers
print("🔎 Transformers version:", transformers.__version__)
print("🔎 Transformers path:", transformers.__file__)


# =========================
# 1. 数据预处理辅助函数
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
    rows = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_tokens = 0
    sampled = []
    for _, row in rows.iterrows():
        quote = row["quote"]

        if pd.isna(quote) or not str(quote).strip():
            continue

        quote = str(quote).strip()
        try:
            n_tokens = len(tokenizer.tokenize(quote))
        except:
            continue

        if total_tokens + n_tokens > max_tokens and len(sampled) > 0:
            break

        sampled.append(row)
        total_tokens += n_tokens

    print(f"采样类别 {df.iloc[0]['label']} → {len(sampled)} 句，~{total_tokens} tokens")
    return pd.DataFrame(sampled)


def prepare_datasets(csv_path, tokenizer, max_tokens=5000):
    df = pd.read_csv(csv_path)

    df["label"] = df["character"].apply(map_character)

    labels = sorted(df["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # 按类别做 token-based balanced sampling
    sampled_dfs = []
    for label in labels:
        part = df[df["label"] == label]
        sampled_dfs.append(sample_by_tokens(part, tokenizer, max_tokens))

    df_balanced = pd.concat(sampled_dfs).reset_index(drop=True)

    # ===========
    # 80/10/10 划分
    # ===========
    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=0.2,  # 先分出 20%（val+test）
        stratify=df_balanced["label_id"],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 20% → 10% val + 10% test
        stratify=temp_df["label_id"],
        random_state=42
    )

    print(f"📊 数据划分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 转换为 HF Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 正确重命名 label 列
    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset = val_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    return train_dataset, val_dataset, test_dataset, label2id, id2label


# =========================
# 2. Tokenizer 处理
# =========================
def tokenize(batch, tokenizer):
    return tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# 3. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# =========================
# 4. 主训练流程
# =========================

def main():
    model_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 处理数据
    train_dataset, val_dataset, test_dataset, label2id, id2label = prepare_datasets(
        "lines/train_lines.csv", tokenizer, max_tokens=5000
    )

    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    # 去掉多余列
    remove_cols = ["no.", "character", "quote", "label"]
    train_dataset = train_dataset.remove_columns(remove_cols)
    val_dataset = val_dataset.remove_columns(remove_cols)
    test_dataset = test_dataset.remove_columns(remove_cols)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./bert-classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 训练
    trainer.train()

    # 保存模型
    trainer.save_model("./bert-classifier")
    tokenizer.save_pretrained("./bert-classifier")

    # =============
    # Test Set Evaluate
    # =============
    print("\n===============================")
    print("📌 Evaluating on TEST SET ...")
    print("===============================")
    test_results = trainer.evaluate(test_dataset)
    print("✅ Test results:", test_results)

    # =========================
    # 交互式 pipeline 测试
    # =========================
    classifier = pipeline("text-classification", model="./bert-classifier", tokenizer=tokenizer)

    print("\n模型训练完成。输入台词进行预测（输入 exit 退出）\n")

    while True:
        text = input("台词: ")
        if text.strip().lower() == "exit":
            break
        pred = classifier(text, truncation=True, max_length=128)
        print("预测结果:", pred)


if __name__ == "__main__":
    main()
