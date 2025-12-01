# llama_extract_single_role_tf.py
# LLaMA-3.2-3b-instruct: extract quotes for a single character
# No langchain dependency, uses transformers pipeline directly

import os
import csv
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === 1. Load API key (if needed) ===
load_dotenv("nvidia_key_3b.env")
NVIDIA_KEY = os.getenv("NVIDIA_KEY_3B")  # not used here but can be used if needed

# === 2. Model initialization ===
model_name = "meta/llama-3.2-3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.0,
    repetition_penalty=1.1
)

# === 3. Parameters: change ROLE for each run ===
ROLE = "holmes"  # e.g., "holmes", "poirot", "marple"
INPUT_DIR = f"_novels/{ROLE}"  # folder containing .txt for this role
OUTPUT_FILE = f"lines/llm_data/{ROLE}_lines.csv"

# === 4. Prompt template ===
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

# === 5. Function: process a single text chunk ===
def extract_quotes_from_text(text, chunk_size=1500):
    """
    Split the text into chunks to avoid context overflow.
    Returns a list of dict: [{'character': ..., 'quote': ...}, ...]
    """
    lines = text.split("\n")
    quotes = []

    chunk = []
    for line in lines:
        chunk.append(line)
        if len(" ".join(chunk).split()) >= chunk_size:
            prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
            output = llm(prompt)[0]["generated_text"]  # transformers pipeline returns list of dict
            quotes.extend(parse_llm_output(output))
            chunk = []

    if chunk:
        prompt = PROMPT_TEMPLATE.format(text="\n".join(chunk))
        output = llm(prompt)[0]["generated_text"]
        quotes.extend(parse_llm_output(output))

    return quotes

# === 6. Parse LLaMA output ===
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

# === 7. Process folder ===
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

    # Save all quotes for this role
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["number", "quote"])
        for i, q in enumerate(all_quotes, 1):
            writer.writerow([i, q["quote"]])

    print(f"Saved {len(all_quotes)} lines for {ROLE} -> {output_file}")

if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_FILE)
