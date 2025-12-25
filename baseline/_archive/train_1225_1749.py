"""
train.py

DISMISSED. 需要增加diagnosis

Train a character identification classifier on prepared dialogue data.

This script defines the TASK (classification setup), not the data cleaning.
Character collapsing (e.g., Watson + Hastings -> Others) is done HERE.

Supported setups:
- 3class: Holmes / Poirot / Marple
- 4class: Holmes / Poirot / Marple / Others

----------
TO RUN:

python train.py --setup 3class --data_path baseline/train_lines_clean_balanced_3class.csv
python train.py --setup 4class --data_path baseline/train_lines_clean_balanced_4class.csv



"""

import argparse
import os
import random
import pandas as pd
import numpy as np

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
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

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =========================
# Label mapping (TASK LOGIC)
# =========================

def apply_label_mapping(df: pd.DataFrame, setup: str) -> pd.DataFrame:
    """
    Apply task-specific label mapping.

    This is where we define what the classifier is asked to do.
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
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# =========================
# Main
# =========================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    df = pd.read_csv(args.data_path)

    print("Applying label mapping...")
    df = apply_label_mapping(df, args.setup)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    df["labels"] = df["label"].map(label2id)

    # =========================
    # Train / Val / Test split
    # =========================

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

    def to_dataset(df):
        return Dataset.from_pandas(df[["quote", "labels"]])

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

    training_args = TrainingArguments(
        output_dir=f"./models/{args.setup}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()

    print("Evaluating on test set...")
    results = trainer.evaluate(test_ds)
    print(results)


if __name__ == "__main__":
    main()
