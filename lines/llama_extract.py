# llama_extract.py
# LLaMA-3.2-3b-instruct (NVIDIA) extract quotes for a single character (English)
# Uses ChatNVIDIA from llm_config.py, no Hugging Face
# Per file proceeding, safe resume
# 12.07

import os
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_config import get_llama_llm

# === 1. Parameters ===
ROLE = "holmes" # Change to "poirot" or "marple" or "holmes" per run
INPUT_DIR = f"_novels/{ROLE}"
OUTPUT_DIR = f"lines/llm_data/{ROLE}"  # << change directory structure

# === 2. Prompt template (English) ===
PROMPT_TEMPLATE = f"""
You are a novel text analysis assistant. Please extract dialogue lines spoken by the character {ROLE} from the text below.

Output format: CSV, each line is one quote:
character,quote

Requirements:
- Keep the original text exactly as in the novel
- Do NOT create or invent quotes
- Do NOT output unrelated text

Text:
{{text}}
"""

# === 3. Initialize NVIDIA LLM ===
llm = get_llama_llm(model_name="meta/llama-3.2-3b-instruct")

# === 4. Functions ===
def extract_quotes_from_text(text, chunk_size=1500):
    """
    Split the text into chunks to avoid context overflow.
    Returns a list of dicts: [{'character': ..., 'quote': ...}, ...]
    """
    lines = text.split("\n")
    quotes = []

    chunk = []
    for line in lines:
        chunk.append(line)
        if len(" ".join(chunk).split()) >= chunk_size:
            prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
            output = call_nvidia_llm(prompt)
            quotes.extend(parse_llm_output(output))
            chunk = []

    if chunk:
        prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
        output = call_nvidia_llm(prompt)
        quotes.extend(parse_llm_output(output))

    return quotes

def call_nvidia_llm(prompt):
    msg = llm.invoke(prompt)
    if hasattr(msg, "content"):
        return msg.content
    return str(msg)



def parse_llm_output(text):
    """
    Convert LLaMA output to list of dicts [{'character':..., 'quote':...}]
    Assumes CSV-like output: character,quote per line
    """
    result = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        if "," not in line:
            continue
        character, quote = line.split(",", 1)
        result.append({"character": character.strip(), "quote": quote.strip()})
    return result

# === 5. Per-file processing (resume-safe) ===
def process_file(input_path, output_path):
    """
    Process ONE novel file → one CSV.
    Skip if already processed.
    """

    # if output already exists → skip
    if os.path.exists(output_path):
        print(f"[skip] {os.path.basename(input_path)} already processed.")
        return

    print(f"[run] Processing {os.path.basename(input_path)} ...")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    quotes = extract_quotes_from_text(text)

    # Save per-file CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["number", "quote"])
        for i, q in enumerate(quotes, 1):
            writer.writerow([i, q["quote"]])

    print(f"[done] Saved {len(quotes)} lines -> {output_path}")

# === 6. Process all txt files ===
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", ".csv"))

        process_file(input_path, output_path)


# === 7. Main ===
if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)