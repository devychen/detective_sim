"""
train.py

Train a character identification classifier on prepared dialogue data.

 holmes -> 0
 marple -> 1
 poirot -> 2


【新增功能说明（重要）】
--------------------------------------------------
1. 自动 checkpoint 保存（每个 epoch 保存一次）
2. 自动查找最新 checkpoint，支持中断后恢复训练
3. 支持“增量训练”：
   - --epochs 表示“本次额外训练多少轮”
   - 如果已有 checkpoint，会在原基础上继续
4. 轻量级保存：
   - 只保留最近 N 个 checkpoint（默认 3 个）
   - 避免频繁 I/O，不影响训练速度
--------------------------------------------------

TO RUN:

# 第一次训练（从零开始）
python baseline/train.py --setup 3class --data_path baseline/train_lines_clean_balanced_3class.csv --epochs 3

# 中断后继续训练（自动从最新 checkpoint 继续，再训练 2 个 epoch）
python baseline/train.py --setup 3class --data_path baseline/train_lines_clean_balanced_3class.csv  --epochs 2


"""

import argparse
import os
import random
import numpy as np
import pandas as pd

import sys
from datetime import datetime


from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# =========================
# Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setup",
        type=str,
        choices=["3class", "4class"],
        required=True,
        help="Classification setup"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to prepared CSV dataset"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-cased"
    )

    # epochs = 本次要“再训练”多少轮（支持增量）
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================
# Task definition (LABEL LOGIC)
# =========================

def apply_label_mapping(df: pd.DataFrame, setup: str) -> pd.DataFrame:
    """
    定义模型要学的分类任务。

    3class:
        holmes / poirot / marple

    4class:
        holmes / poirot / marple / others
    """
    if setup == "3class":
        df = df[df["character"].isin(["holmes", "poirot", "marple"])]
        df["label"] = df["character"]

    elif setup == "4class":
        def map_label(c):
            if c in {"holmes", "poirot", "marple"}:
                return c
            else:
                return "others"

        df["label"] = df["character"].apply(map_label)

    return df.reset_index(drop=True)


# =========================
# Tokenization
# =========================

def tokenize(batch, tokenizer):
    """
    将文本转成 BERT 可以接受的 token 格式
    """
    return tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# Metrics (during training)
# =========================

def compute_metrics(eval_pred):
    """
    训练/验证阶段实时监控的指标
    """
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# =========================
# Helper: 自动寻找最新 checkpoint
# =========================

def find_latest_checkpoint(output_dir):
    """
    在 output_dir 中查找最新的 checkpoint

    checkpoint 命名形如：
        checkpoint-500
        checkpoint-1000

    返回：
        - 最新 checkpoint 路径
        - 如果不存在，返回 None
    """
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint")
    ]

    if not checkpoints:
        return None

    # 按修改时间排序，取最新的
    return max(checkpoints, key=os.path.getmtime)


# =========================
# Main
# =========================

def main():
    args = parse_args()

    # -------------------------
    # Reproducibility
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)

    print("Applying task-specific label mapping...")
    df = apply_label_mapping(df, args.setup)

    # 明确打乱，防止顺序偏差
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Label 映射
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Label mapping:")
    for k, v in label2id.items():
        print(f"  {k} -> {v}")

    df["labels"] = df["label"].map(label2id)

    # -------------------------
    # Train / Val / Test split
    # -------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["labels"],
        random_state=args.seed
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["labels"],
        random_state=args.seed
    )

    def to_dataset(frame):
        return Dataset.from_pandas(frame[["quote", "labels"]])

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # -------------------------
    # TrainingArguments（关键修改部分）
    # -------------------------
    output_dir = f"./models/{args.setup}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,

        # 每个 epoch 做一次验证
        eval_strategy="epoch",

        # 每个 epoch 保存一次 checkpoint（避免频繁 I/O）
        save_strategy="epoch",

        # 最多保留 3 个 checkpoint（轻量级）
        save_total_limit=3,

        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # 注意：这是“本次”要训练的 epoch 数
        num_train_epochs=args.epochs,

        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -------------------------
    # 自动断点续训
    # -------------------------
    latest_checkpoint = find_latest_checkpoint(output_dir)

    if latest_checkpoint:
        print(f"Loading model weights from {latest_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
        latest_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
        )
    else:
        print("\nNo checkpoint found, training from scratch")

    print("\nTraining model...")
    trainer.train(resume_from_checkpoint=None)

    # =========================
    # Diagnostics
    # =========================

    print("\nEvaluating on TRAIN set (diagnostic)...")
    train_preds = trainer.predict(train_ds)
    train_y_true = train_preds.label_ids
    train_y_pred = train_preds.predictions.argmax(axis=1)
    print(f"Train accuracy: {accuracy_score(train_y_true, train_y_pred):.4f}")

    print("\nEvaluating on TEST set...")
    test_preds = trainer.predict(test_ds)
    y_true = test_preds.label_ids
    y_pred = test_preds.predictions.argmax(axis=1)

    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Test macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nPer-class report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in sorted(id2label)]
    ))


if __name__ == "__main__":
    main()
