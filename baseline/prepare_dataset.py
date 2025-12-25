"""
prepare_dataset.py

CURRENT

This script prepares a clean, balanced dataset for training character
identification classifiers based on LLM-extracted dialogue.

Parameterised version（--classes 3|4）

IMPORTANT DESIGN PRINCIPLE
--------------------------
This script performs ONLY data-level preprocessing.

It does NOT:
- merge characters into task-specific labels (e.g., "others")
- assume a particular classification setup

All label collapsing (e.g., watson + hastings -> others) should be done
at TRAINING time, not here.

SUPPORTED MODES
---------------
--classes 3 : keep only the three main detectives
--classes 4 : keep main detectives + secondary characters
              (to be merged into "others" later during training)

TO RUN
---------------
python prepare_dataset.py --classes 3
python prepare_dataset.py --classes 4

"""

import os
import glob
import argparse
import random
import string
import pandas as pd
from transformers import AutoTokenizer


# =========================
# Argument Parsing
# =========================

def parse_args():
    """
    Parse command-line arguments.

    --classes determines which characters are INCLUDED
    in the prepared dataset.

    Note:
    - This does NOT define the final classification labels.
    - It only controls which utterances are kept.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        type=int,
        choices=[3, 4],
        required=True,
        help="3 = main detectives only; 4 = include secondary characters (for 'others')"
    )
    return parser.parse_args()


# =========================
# Configuration (DO NOT HARD-CODE TASK LOGIC HERE)
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "lines", "llm_data")

TOKENIZER_NAME = "google-bert/bert-base-cased"
MIN_TOKENS = 3
RANDOM_SEED = 42


# Token budgets per character
TOKEN_BUDGET_3CLASS = 120_000
TOKEN_BUDGET_4CLASS = 50_000


# =========================
# Utility Functions
# =========================

def is_noise(text: str) -> bool:
    """
    Determine whether a quote is likely to contain no useful linguistic signal.

    Conservative definition:
    - empty or missing
    - placeholder tokens ("Quote")
    - punctuation-only strings

    We intentionally avoid aggressive normalization so as not to
    remove stylistic information.
    """
    if not isinstance(text, str):
        return True

    stripped = text.strip()

    if stripped == "":
        return True

    if stripped.lower() in {"quote", "\"quote\"", "''", "\"\""}:
        return True

    no_punct = stripped.translate(str.maketrans("", "", string.punctuation))
    if no_punct.strip() == "":
        return True

    return False


def load_all_llm_data(data_root: str, allowed_characters: list) -> pd.DataFrame:
    """
    Load all LLM-extracted dialogue CSV files into a single DataFrame.

    Expected directory structure:
        llm_data/
          ├── character_name/
          │     ├── book1.csv
          │     ├── book2.csv

    Output columns:
        - quote
        - character
        - source_book
    """
    rows = []

    for character in allowed_characters:
        char_dir = os.path.join(data_root, character)

        if not os.path.isdir(char_dir):
            print(f"WARNING: directory not found for '{character}', skipping.")
            continue

        csv_files = glob.glob(os.path.join(char_dir, "*.csv"))

        for csv_path in csv_files:
            book_name = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                rows.append({
                    "quote": row.get("quote", ""),
                    "character": character,
                    "source_book": book_name,
                })

    return pd.DataFrame(rows)


def add_token_counts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """
    Compute token counts for each utterance.

    Token counts are used ONLY for:
    - removing very short utterances
    - balancing data by token exposure

    They are NOT used for model training here.
    """
    df["n_tokens"] = df["quote"].apply(
        lambda x: len(tokenizer.tokenize(x)) if isinstance(x, str) else 0
    )
    return df


# =========================
# Cleaning Steps
# =========================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply conservative cleaning:

    1. Remove noise / placeholders
    2. Remove very short utterances
    3. Remove exact duplicates per character

    No stylistic normalization is applied.
    """
    original_size = len(df)

    df = df[~df["quote"].apply(is_noise)]
    df = df[df["n_tokens"] >= MIN_TOKENS]
    df = df.drop_duplicates(subset=["character", "quote"])

    print(f"Cleaning reduced dataset from {original_size} to {len(df)} utterances.")
    return df


# =========================
# Balancing Step
# =========================

def balance_by_token_budget(df: pd.DataFrame, token_budget: int) -> pd.DataFrame:
    """
    Balance the dataset across characters using a token-based budget.

    For each character:
    - shuffle utterances
    - accumulate utterances until token budget is reached

    This avoids dominance by characters with more dialogue or longer turns.
    """
    balanced_parts = []

    for character, part in df.groupby("character"):
        part = part.sample(frac=1, random_state=RANDOM_SEED)

        total_tokens = 0
        selected = []

        for _, row in part.iterrows():
            if total_tokens + row["n_tokens"] > token_budget:
                break
            selected.append(row)
            total_tokens += row["n_tokens"]

        print(
            f"Character '{character}': "
            f"{len(selected)} utterances, {total_tokens} tokens"
        )

        balanced_parts.append(pd.DataFrame(selected))

    return pd.concat(balanced_parts).reset_index(drop=True)


# =========================
# Main Pipeline
# =========================

def main():
    args = parse_args()
    random.seed(RANDOM_SEED)

    # Define which characters are INCLUDED (not merged!)
    if args.classes == 3:
        included_characters = ["holmes", "poirot", "marple"]
        token_budget = TOKEN_BUDGET_3CLASS
        output_name = "train_lines_clean_balanced_3class.csv"

    else:  # args.classes == 4
        included_characters = ["holmes", "poirot", "marple", "watson", "hastings"]
        token_budget = TOKEN_BUDGET_4CLASS
        output_name = "train_lines_clean_balanced_4class.csv"

    output_path = os.path.join(PROJECT_ROOT, "baseline", output_name)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading LLM-extracted data...")
    df = load_all_llm_data(DATA_ROOT, included_characters)
    print(f"Loaded {len(df)} utterances.")

    print("Computing token counts...")
    df = add_token_counts(df, tokenizer)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Balancing dataset by token budget...")
    df = balance_by_token_budget(df, token_budget)

    final_df = df[["quote", "character"]]

    print(f"Saving final dataset to: {output_path}")
    final_df.to_csv(output_path, index=False)

    print("Dataset preparation complete.")
    print("NOTE: No character merging was performed.")
    print("      Label collapsing should be done during training.")


if __name__ == "__main__":
    main()
