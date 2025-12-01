# llama_extract_single_role_nvidia.py
# LLaMA-3.2-3b-instruct (NVIDIA) extract quotes for a single character (English)
# Uses ChatNVIDIA from llm_config.py, no Hugging Face

import os
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_config import get_llama_llm

# === 1. Parameters ===
ROLE = "holmes"  # Change to "poirot" or "marple" per run
INPUT_DIR = f"_novels/{ROLE}"  # folder containing .txt for this role
OUTPUT_FILE = f"lines/cleaned_{ROLE}_lines.csv"

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
    """
    Call NVIDIA LLM via ChatNVIDIA
    """
    response = llm.chat(messages=[{"role": "user", "content": prompt}])
    return response.message

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

# === 5. Process folder ===
def process_directory(input_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_quotes = []

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".txt"):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"Processing {filename} ...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        quotes = extract_quotes_from_text(text)
        all_quotes.extend(quotes)

    # Save CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["number", "quote"])
        for i, q in enumerate(all_quotes, 1):
            writer.writerow([i, q["quote"]])

    print(f"Saved {len(all_quotes)} lines for {ROLE} -> {output_file}")

# === 6. Main ===
if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_FILE)
