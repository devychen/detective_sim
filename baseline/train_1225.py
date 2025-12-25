"""
train.py

This script trains a supervised character identification classifier
on LLM-extracted dialogue data.

The classifier is intended as an *upper bound* on character identifiability:
if a standard supervised model cannot reliably distinguish characters
based on their dialogue, this provides important context for later analyses
(e.g., role consistency and OOC behavior in LLM agents).

The script performs the following steps:

1. Load a clean, balanced CSV dataset
2. Encode character labels
3. Split data into train / validation / test sets (stratified)
4. Tokenize utterances
5. Train a BERT-based sequence classifier
6. Evaluate performance on validation and test sets
7. Report accuracy, macro-F1, and confusion matrix

No hyperparameter tuning or advanced optimization is performed by design.
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Configuration
# =========================

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "baseline", "train_lines_clean_balanced.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "baseline", "bert_character_classifier")

# Model
MODEL_NAME = "google-bert/bert-base-cased"
MAX_LENGTH = 128  # justified by prior diagnostics

# Training
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5


# =========================
# Utility Functions
# =========================

def load_and_encode_data(csv_path: str):
    """
    Load the training CSV and encode character labels.

    Returns:
        df: pandas DataFrame with an added 'label_id' column
        label2id: mapping from character string to integer ID
        id2label: inverse mapping
    """
    df = pd.read_csv(csv_path)

    labels = sorted(df["character"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["label_id"] = df["character"].map(label2id)

    return df, label2id, id2label


def split_dataset(df: pd.DataFrame):
    """
    Split the dataset into train / validation / test sets.

    Splitting is stratified by label to ensure equal class proportions.

    Split ratios:
        - 80% train
        - 10% validation
        - 10% test
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label_id"],
        random_state=RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_id"],
        random_state=RANDOM_SEED,
    )

    return train_df, val_df, test_df


def tokenize_batch(batch, tokenizer):
    """
    Tokenize a batch of utterances.

    Truncation and padding are applied to a fixed maximum length,
    which has been justified via prior length diagnostics.
    """
    return tokenizer(
        batch["quote"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the classifier.

    Metrics:
        - Accuracy
        - Macro-averaged F1 (important for class-balanced evaluation)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Plot and save a confusion matrix for qualitative error analysis.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================
# Main Training Pipeline
# =========================

def main():
    print("Loading and encoding dataset...")
    df, label2id, id2label = load_and_encode_data(DATA_PATH)

    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)

    print(
        f"Dataset sizes: "
        f"train={len(train_df)}, "
        f"val={len(val_df)}, "
        f"test={len(test_df)}"
    )

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing datasets...")
    train_ds = train_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)

    # Rename label column to match Trainer expectations
    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    # Remove unused columns
    columns_to_remove = ["quote", "character"]
    train_ds = train_ds.remove_columns(columns_to_remove)
    val_ds = val_ds.remove_columns(columns_to_remove)
    test_ds = test_ds.remove_columns(columns_to_remove)

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    print("Test results:", test_results)

    # Confusion matrix
    print("Generating confusion matrix...")
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=[id2label[i] for i in range(len(id2label))],
        output_path=cm_path,
    )

    print(f"Confusion matrix saved to {cm_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
