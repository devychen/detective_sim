"""
data_diagnostics.py

This script performs systematic diagnostics on LLM-extracted dialogue data
before any classifier training is attempted.

The goal is to answer the following research questions:

1. How large is the dataset per character (utterances, tokens)?
2. How balanced are the datasets across characters?
3. How long are the extracted dialogue units?
4. How much information would be lost by truncating inputs at 128 / 256 / 512 tokens?
5. How noisy is the data (empty lines, placeholders, very short utterances)?
6. Is there evidence of LLM-induced repetition or templating?

This script is intentionally model-agnostic and does NOT train any classifier.

12.13
"""

import os
import glob
import pandas as pd
import numpy as np
import string
from collections import Counter
from transformers import AutoTokenizer


# =========================
# Configuration
# =========================

DATA_ROOT = "lines/llm_data"
MODEL_NAME = "google-bert/bert-base-cased"

TOKEN_THRESHOLDS = [128, 256, 512]
SHORT_UTTERANCE_TOKENS = 3


# =========================
# Helper Functions
# =========================

def is_empty_or_noise(text: str) -> bool:
    """
    Identify utterances that likely contain no linguistic signal.

    This includes:
    - empty strings
    - placeholder tokens such as "Quote"
    - strings consisting only of punctuation or quotes
    """
    if not isinstance(text, str):
        return True

    stripped = text.strip()

    if stripped == "":
        return True

    if stripped.lower() in {"quote", "\"quote\"", "''", "\"\""}:
        return True

    # Remove punctuation and check if anything remains
    no_punct = stripped.translate(str.maketrans("", "", string.punctuation))
    if no_punct.strip() == "":
        return True

    return False


def load_llm_data(root_dir: str) -> pd.DataFrame:
    """
    Traverse the llm_data directory structure and load all CSV files.

    Expected structure:
        llm_data/
          ├── holmes/
          │     ├── book1.csv
          │     ├── book2.csv
          ├── marple/
          ├── poirot/
          ├── others/

    Returns a unified DataFrame with explicit metadata columns:
        - quote
        - character
        - source_book
    """
    all_rows = []

    for character in sorted(os.listdir(root_dir)):
        char_dir = os.path.join(root_dir, character)

        if not os.path.isdir(char_dir):
            continue

        csv_files = glob.glob(os.path.join(char_dir, "*.csv"))

        for csv_path in csv_files:
            book_name = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                all_rows.append({
                    "quote": row.get("quote", ""),
                    "character": character,
                    "source_book": book_name
                })

    master_df = pd.DataFrame(all_rows)
    return master_df


def add_token_statistics(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """
    Compute token-level statistics using a transformer tokenizer.

    Adds the following columns:
        - n_tokens
        - n_words
    """
    token_lengths = []
    word_lengths = []

    for text in df["quote"]:
        if isinstance(text, str):
            token_lengths.append(len(tokenizer.tokenize(text)))
            word_lengths.append(len(text.split()))
        else:
            token_lengths.append(0)
            word_lengths.append(0)

    df["n_tokens"] = token_lengths
    df["n_words"] = word_lengths

    return df


# =========================
# Diagnostic Reports
# =========================

def report_dataset_size(df: pd.DataFrame):
    """
    Report dataset size and balance per character.
    """
    print("\n===============================")
    print("DATASET SIZE & BALANCE")
    print("===============================")

    for character, part in df.groupby("character"):
        print(f"\nCharacter: {character}")
        print(f"  Utterances: {len(part)}")
        print(f"  Total tokens: {part['n_tokens'].sum():,}")
        print(f"  Avg tokens / utterance: {part['n_tokens'].mean():.2f}")


def report_length_distribution(df: pd.DataFrame):
    """
    Report token length distribution overall and per character,
    with special focus on truncation thresholds.
    """
    print("\n===============================")
    print("LENGTH DISTRIBUTION & TRUNCATION RISK")
    print("===============================")

    def summarize(part: pd.DataFrame, label: str):
        print(f"\n[{label}]")
        print(f"  Mean tokens: {part['n_tokens'].mean():.2f}")
        print(f"  Median tokens: {part['n_tokens'].median():.2f}")
        print(f"  90th percentile: {np.percentile(part['n_tokens'], 90):.2f}")
        print(f"  Max tokens: {part['n_tokens'].max()}")

        for t in TOKEN_THRESHOLDS:
            pct = (part["n_tokens"] > t).mean() * 100
            print(f"  > {t} tokens: {pct:.2f}%")

    summarize(df, "OVERALL")

    for character, part in df.groupby("character"):
        summarize(part, character)


def report_noise_statistics(df: pd.DataFrame):
    """
    Report how much of the data is likely noisy or uninformative.
    """
    print("\n===============================")
    print("NOISE & DATA QUALITY CHECK")
    print("===============================")

    df["is_noise"] = df["quote"].apply(is_empty_or_noise)
    df["is_short"] = df["n_tokens"] < SHORT_UTTERANCE_TOKENS

    for character, part in df.groupby("character"):
        print(f"\nCharacter: {character}")
        print(f"  Empty / noise utterances: {part['is_noise'].mean() * 100:.2f}%")
        print(f"  Very short (<{SHORT_UTTERANCE_TOKENS} tokens): {part['is_short'].mean() * 100:.2f}%")


def report_duplicates(df: pd.DataFrame):
    """
    Report exact duplicate utterances, which can occur with LLM extraction.
    """
    print("\n===============================")
    print("DUPLICATION CHECK")
    print("===============================")

    for character, part in df.groupby("character"):
        total = len(part)
        duplicates = part["quote"].duplicated().sum()
        print(f"\nCharacter: {character}")
        print(f"  Duplicate utterances: {duplicates} ({duplicates / total * 100:.2f}%)")


# =========================
# Main Execution
# =========================

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading LLM-extracted dialogue data...")
    df = load_llm_data(DATA_ROOT)

    print(f"Total utterances loaded: {len(df)}")

    print("Computing token statistics...")
    df = add_token_statistics(df, tokenizer)

    # Run diagnostics
    report_dataset_size(df)
    report_length_distribution(df)
    report_noise_statistics(df)
    report_duplicates(df)

    print("\nDiagnostics complete.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()
