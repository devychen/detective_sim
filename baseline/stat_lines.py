import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("lines/train_lines.csv")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

stats = {}

for char in df["character"].unique():
    part = df[df["character"] == char]
    token_lens = part["quote"].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))

    stats[char] = {
        "num_sentences": len(part),
        "total_tokens": token_lens.sum(),
        "avg_tokens": token_lens.mean(),
        "max_tokens": token_lens.max(),
        "pct_over_128": (token_lens > 128).mean(),
        "pct_over_256": (token_lens > 256).mean(),
    }

import print
print.pprint(stats)
