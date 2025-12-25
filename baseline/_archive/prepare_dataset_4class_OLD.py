"""
prepare_dataset.py

5 characters. 该.py暂时遗弃。DISMISSED.

This script prepares a clean, balanced dataset for training a character
identification classifier based on LLM-extracted dialogue.

It performs the following steps, in order:

1. Load all CSV files from a structured directory of characters and books
2. Remove noisy or uninformative utterances
3. Remove very short utterances
4. Remove exact duplicate utterances per character
5. Balance the dataset across characters using a token-based budget
6. Output a single clean CSV file ready for classifier training

IMPORTANT:
- This script does NOT train any model
- All preprocessing steps are intentionally conservative and interpretable
"""

import os
import glob
import random
import string
import pandas as pd
from transformers import AutoTokenizer


# =========================
# Configuration (SAFE TO EDIT)
# =========================

# Path settings (relative to this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "lines", "llm_data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "baseline", "train_lines_clean_balanced.csv")

# Characters to INCLUDE (explicit on purpose)
INCLUDED_CHARACTERS = [
    "holmes",
    "poirot",
    "marple",
    "watson",
    "hastings",
]

# Token budget per character (after cleaning & deduplication)
TOKEN_BUDGET_PER_CHARACTER = 50_000

# Minimum length requirement
MIN_TOKENS = 3

# Tokenizer for token counting (NOT for training here)
TOKENIZER_NAME = "google-bert/bert-base-cased"

# Random seed for reproducibility
RANDOM_SEED = 42


# =========================
# Utility Functions
# =========================

def is_noise(text: str) -> bool:
    """
    Determine whether a quote is likely to contain no useful linguistic signal.

    We consider a quote as noise if:
    - it is empty or missing
    - it is a placeholder such as "Quote"
    - it consists only of punctuation or quotation marks

    This function is intentionally conservative: it only removes
    utterances that are extremely unlikely to be informative.
    """
    if not isinstance(text, str):
        return True

    stripped = text.strip()

    if stripped == "":
        return True

    if stripped.lower() in {"quote", "\"quote\"", "''", "\"\""}:
        return True

    # Remove punctuation; if nothing remains, it's noise
    no_punct = stripped.translate(str.maketrans("", "", string.punctuation))
    if no_punct.strip() == "":
        return True

    return False


def load_all_llm_data(data_root: str, allowed_characters: list) -> pd.DataFrame:
    """
    Load all LLM-extracted dialogue CSV files into a single DataFrame.

    Directory structure is assumed to be:
        llm_data/
          ├── character_name/
          │     ├── book1.csv
          │     ├── book2.csv

    Only characters listed in `allowed_characters` are loaded.

    The resulting DataFrame contains:
        - quote: the dialogue text
        - character: character label
        - source_book: which book the line came from
    """
    rows = []

    for character in allowed_characters:
        char_dir = os.path.join(data_root, character)

        if not os.path.isdir(char_dir):
            print(f"WARNING: directory not found for character '{character}', skipping.")
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
    Compute token counts for each utterance using a transformer tokenizer.

    This token count is used ONLY for:
    - filtering short utterances
    - balancing the dataset by token budget

    It is not used for training in this script.
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
    Apply conservative cleaning steps to the dataset:

    1. Remove empty / placeholder / punctuation-only utterances
    2. Remove utterances shorter than MIN_TOKENS
    3. Remove exact duplicate quotes within each character

    No normalization (lowercasing, stemming, etc.) is applied,
    as stylistic information should be preserved.
    """
    original_size = len(df)

    # Remove noise
    df = df[~df["quote"].apply(is_noise)]

    # Remove very short utterances
    df = df[df["n_tokens"] >= MIN_TOKENS]

    # Remove exact duplicates per character
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
    - Shuffle utterances
    - Add utterances until the cumulative token count
      reaches (but does not exceed too much) the given budget

    This avoids biasing the classifier toward characters
    with more or longer utterances.
    """
    balanced_parts = []

    for character, part in df.groupby("character"):
        part = part.sample(frac=1, random_state=RANDOM_SEED)  # shuffle
        total_tokens = 0
        selected_rows = []

        for _, row in part.iterrows():
            if total_tokens + row["n_tokens"] > token_budget:
                break

            selected_rows.append(row)
            total_tokens += row["n_tokens"]

        print(
            f"Character '{character}': "
            f"{len(selected_rows)} utterances, "
            f"{total_tokens} tokens"
        )

        balanced_parts.append(pd.DataFrame(selected_rows))

    return pd.concat(balanced_parts).reset_index(drop=True)


# =========================
# Main Pipeline
# =========================

def main():
    random.seed(RANDOM_SEED)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading LLM-extracted data...")
    df = load_all_llm_data(DATA_ROOT, INCLUDED_CHARACTERS)
    print(f"Loaded {len(df)} utterances.")

    print("Computing token counts...")
    df = add_token_counts(df, tokenizer)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Balancing dataset by token budget...")
    df = balance_by_token_budget(df, TOKEN_BUDGET_PER_CHARACTER)

    # Keep only columns needed for training
    final_df = df[["quote", "character"]]

    print(f"Saving final dataset to: {OUTPUT_PATH}")
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset preparation complete.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()

"""
得到结果：

Loading tokenizer...
Loading LLM-extracted data...
Loaded 26983 utterances.
Computing token counts...
Token indices sequence length is longer than the specified maximum sequence length for this model (522 > 512). Running this sequence through the model will result in indexing errors
Cleaning dataset...
Cleaning reduced dataset from 26983 to 22298 utterances.
Balancing dataset by token budget...
Character 'hastings': 2099 utterances, 49944 tokens
Character 'holmes': 1849 utterances, 49952 tokens
Character 'marple': 1807 utterances, 49998 tokens
Character 'poirot': 1998 utterances, 49825 tokens
Character 'watson': 1824 utterances, 49863 tokens
Saving final dataset to: /Users/ychen/Documents/detective_sim/lines/train_lines_clean_balanced.csv
Dataset preparation complete.
No model training was performed.

该py只是用 tokenizer 数 token，并没有把 token 喂给模型，更没有 forward pass。所以中间的warning没关系。

从你的统计结果来看，这条 warning 甚至是“好消息”。
你的 diagnostics 里已经显示：512 tokens：0.01%。现在 preprocessing 里又再次印证：偶尔存在极长 quote（522, 1224 tokens）
这说明你的 pipeline 是诚实的，没有偷偷 truncate，而不是 tokenizer 把长句悄悄截断了。
这在方法论上是加分的。

而之后在训练阶段，你本来就计划：truncation=True，max_length=128（因为大部分length都<128且长 utterances 是“异质样本”）
所以在训练阶段：这个 warning 会自然消失。因为你主动截断了
--> We set the maximum input length to 128 tokens.
This choice is motivated by the empirical length distribution of the extracted utterances: fewer than 2% exceed this threshold.
As a result, truncation introduces negligible information loss while avoiding unnecessary padding and variance.
"""