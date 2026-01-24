"""
train_log.py

Train a character identification classifier on prepared dialogue data.

 holmes -> 0
 marple -> 1
 poirot -> 2


【新增功能说明（重要）】
--------------------------------------------------
在train.py的基础上，唯一的改动是增加了timestamp
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
from datetime import datetime  # ★ NEW
import numpy as np
import pandas as pd

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
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-cased"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================
# Task definition
# =========================

def apply_label_mapping(df: pd.DataFrame, setup: str) -> pd.DataFrame:
    if setup == "3class":
        df = df[df["character"].isin(["holmes", "poirot", "marple"])]
        df["label"] = df["character"]

    elif setup == "4class":
        def map_label(c):
            return c if c in {"holmes", "poirot", "marple"} else "others"
        df["label"] = df["character"].apply(map_label)

    return df.reset_index(drop=True)


# =========================
# Tokenization
# =========================

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# Metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# =========================
# Helper: latest checkpoint
# =========================

def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint")
    ]

    if not checkpoints:
        return None

    return max(checkpoints, key=os.path.getmtime)


# =========================
# Main
# =========================

def main():
    args = parse_args()

    # ★ NEW: result file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "baseline"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir, f"train_results_{timestamp}.txt"
    )

    f = open(results_path, "w", encoding="utf-8")

    def log(msg=""):
        f.write(str(msg) + "\n")

    # -------------------------
    # Reproducibility
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    log("Loading dataset...")
    df = pd.read_csv(args.data_path)

    log("Applying task-specific label mapping...")
    df = apply_label_mapping(df, args.setup)

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    log("Label mapping:")
    for k, v in label2id.items():
        log(f"  {k} -> {v}")

    df["labels"] = df["label"].map(label2id)

    # -------------------------
    # Split
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
    # TrainingArguments
    # -------------------------
    output_dir = f"./models/{args.setup}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
    # Resume logic
    # -------------------------
    latest_checkpoint = find_latest_checkpoint(output_dir)

    if latest_checkpoint:
        log(f"Loading model weights from {latest_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            latest_checkpoint,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
    else:
        log("No checkpoint found, training from scratch")

    log("\nTraining model...")
    trainer.train(resume_from_checkpoint=None)

    # -------------------------
    # Diagnostics
    # -------------------------
    log("\nEvaluating on TRAIN set...")
    train_preds = trainer.predict(train_ds)
    train_y_true = train_preds.label_ids
    train_y_pred = train_preds.predictions.argmax(axis=1)
    log(f"Train accuracy: {accuracy_score(train_y_true, train_y_pred):.4f}")

    log("\nEvaluating on TEST set...")
    test_preds = trainer.predict(test_ds)
    y_true = test_preds.label_ids
    y_pred = test_preds.predictions.argmax(axis=1)

    log(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    log(f"Test macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    log("\nConfusion Matrix:")
    log(confusion_matrix(y_true, y_pred))

    log("\nPer-class report:")
    log(classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in sorted(id2label)]
    ))

    f.close()
    print(f"[Saved] {results_path}")


if __name__ == "__main__":
    main()
