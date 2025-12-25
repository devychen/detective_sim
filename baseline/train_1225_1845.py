"""
train.py

Train a character identification classifier on prepared dialogue data.

This script defines the TASK (classification setup), not the data cleaning.
Character collapsing (e.g., Watson + Hastings -> Others) is done HERE.

Supported setups:
- 3class: Holmes / Poirot / Marple
- 4class: Holmes / Poirot / Marple / Others

This script outputs:
- Train accuracy (diagnostic)
- Validation accuracy & macro-F1 (during training)
- Test accuracy & macro-F1
- Confusion matrix
- Per-class precision / recall / F1

IMPORTANT:
- Training data is SHUFFLED (no character-order bias)
- Test set is strictly held-out


---------- 
TO RUN: 

python baseline/train.py --setup 3class --data_path baseline/train_lines_clean_balanced_3class.csv 
python baseline/train.py --setup 4class --data_path baseline/train_lines_clean_balanced_4class.csv

"""

import argparse
import random
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
# Task definition (LABEL LOGIC)
# =========================

def apply_label_mapping(df: pd.DataFrame, setup: str) -> pd.DataFrame:
    """
    Define what the classifier is asked to predict.

    3class:
        holmes / poirot / marple

    4class:
        holmes / poirot / marple / others
        (others includes Watson, Hastings, etc.)
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
# Metrics (used during training)
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# =========================
# Main
# =========================

def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)

    print("Applying task-specific label mapping...")
    df = apply_label_mapping(df, args.setup)

    # Explicit shuffle to avoid any ordering artefacts
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Label mapping:")
    for k, v in label2id.items():
        print(f"  {k} -> {v}")

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

    training_args = TrainingArguments(
        output_dir=f"./models/{args.setup}",
        eval_strategy="epoch",
        save_strategy="epoch",
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

    print("\nTraining model...")
    trainer.train()

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
