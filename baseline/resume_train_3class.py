"""
resume_train_3class.py

Resume training of the 3-class character classifier from an existing checkpoint.

Features:
- Continue training from the last checkpoint
- Light-weight checkpointing (save once per epoch, keep only last 2)
- Supports total epochs parameter
- Preserves existing train/val/test split, shuffle, and evaluation metrics
"""

import argparse
import random
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to prepared 3-class CSV")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-cased")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory of the checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of epochs (including previous)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tokenize(batch, tokenizer):
    return tokenizer(batch["quote"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    df = pd.read_csv(args.data_path)
    df = df[df["character"].isin(["holmes", "poirot", "marple"])]
    df["label"] = df["character"]
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["labels"], random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["labels"], random_state=args.seed)

    def to_dataset(frame):
        return Dataset.from_pandas(frame[["quote", "labels"]])

    train_ds, val_ds, test_ds = to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds = [ds.map(lambda x: tokenize(x, tokenizer), batched=True) for ds in (train_ds, val_ds, test_ds)]
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keep last 2 checkpoints
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

    print("\nResuming training...")
    trainer.train(resume_from_checkpoint=args.checkpoint_dir)

    # Diagnostics
    print("\nEvaluating on TRAIN set...")
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
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)]))


if __name__ == "__main__":
    main()
