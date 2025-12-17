"""
prepare_dataset.py

This script takes LLM-extracted dialogue CSVs and prepares a clean dataset
for classifier training.

Main steps:

1. Load all CSV files for all characters and merge into a master dataframe.
2. Add metadata columns: 'character' and 'source_book'.
3. Clean the data:
    - Remove empty or placeholder quotes
    - Remove extremely short quotes (<3 tokens)
4. Optionally mark duplicate quotes
5. Split dataset into train / val / test sets (stratified by character)
6. Save the cleaned datasets for later use in model training.

"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# Configuration
# =========================

DATA_ROOT = "lines/llm_data"  # Root folder containing character subfolders
MIN_TOKENS = 3           # Minimum number of tokens to keep a quote
TEST_SIZE = 0.1          # Fraction for test set
VAL_SIZE = 0.1           # Fraction for validation set

OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Helper functions
# =========================

def is_empty_or_noise(text):
    """
    Check if a quote is empty or placeholder/noise.
    
    Includes:
        - empty strings
        - placeholder text like "Quote" or '""'
        - strings consisting only of punctuation
    """
    if not isinstance(text, str):
        return True

    stripped = text.strip()

    if stripped == "":
        return True

    if stripped.lower() in {"quote", "\"quote\"", "''", "\"\""}:
        return True

    # Remove punctuation and check if anything remains
    import string
    no_punct = stripped.translate(str.maketrans("", "", string.punctuation))
    if no_punct.strip() == "":
        return True

    return False

def load_llm_data(root_dir):
    """
    Traverse the directory structure and load all CSVs.

    Expected structure:
        llm_data/
          ├── holmes/
          │     ├── book1.csv
          │     ├── book2.csv
          ├── marple/
          ├── poirot/
          ├── others/

    Returns a single dataframe with columns:
        - quote: the dialogue line
        - character: character name
        - source_book: which novel
    """
    all_rows = []

    # Loop through character folders
    for character in sorted(os.listdir(root_dir)):
        char_dir = os.path.join(root_dir, character)
        if not os.path.isdir(char_dir):
            continue

        # Loop through all CSVs in the folder
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

def clean_data(df):
    """
    Clean the dataset by removing noisy and extremely short quotes.
    Also adds a boolean column 'is_duplicate' to mark duplicate quotes.
    """
    # Remove empty / placeholder quotes
    df = df[~df["quote"].apply(is_empty_or_noise)].copy()

    # Remove extremely short quotes (fewer than MIN_TOKENS words)
    df["n_words"] = df["quote"].apply(lambda x: len(str(x).split()))
    df = df[df["n_words"] >= MIN_TOKENS]

    # Mark duplicates
    df["is_duplicate"] = df.duplicated(subset=["quote", "character"])

    return df

def split_dataset(df, test_size=0.1, val_size=0.1, random_state=42):
    """
    Split the cleaned dataset into train / val / test sets.

    Stratification is done based on 'character' to ensure all sets
    have proportional representation of each class.
    """
    # First, split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["character"], random_state=random_state
    )

    # Then split train_val into train and val
    val_relative = val_size / (1 - test_size)  # Adjust fraction
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, stratify=train_val_df["character"], random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# =========================
# Main execution
# =========================

def main():
    print("Loading LLM-extracted dialogue data...")
    df = load_llm_data(DATA_ROOT)
    print(f"Total quotes loaded: {len(df)}")

    print("Cleaning data (removing noise, very short quotes, marking duplicates)...")
    df = clean_data(df)
    print(f"Total quotes after cleaning: {len(df)}")

    print("Splitting dataset into train / val / test...")
    train_df, val_df, test_df = split_dataset(df, test_size=TEST_SIZE, val_size=VAL_SIZE)
    print(f"Train set: {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    print(f"Test set: {len(test_df)}")

    # Save cleaned datasets
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Cleaned datasets saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
