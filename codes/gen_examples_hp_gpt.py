import os
import requests
import json
import dotenv
import datetime
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher

# Load environment variables
dotenv.load_dotenv("openai_key.env")  # Ensure your OpenAI API key is in this file

# API Configuration
MODEL_NAME = "gpt-4"  # or "gpt-3.5-turbo" if you prefer
API_URL = "https://api.openai.com/v1/chat/completions"
OUTPUT_FILE = "hp_examples_05.json"
SIMILARITY_THRESHOLD = 0.9  # 90% similarity threshold for considering duplicates

def calculate_similarity(a: str, b: str) -> float:
    """Calculate the similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_duplicate(new_quote: str, existing_quotes: List[str]) -> bool:
    """Check if a new quote is too similar to any existing quotes"""
    for existing in existing_quotes:
        if calculate_similarity(new_quote, existing) > SIMILARITY_THRESHOLD:
            return True
    return False

def find_examples(prompt_text: str) -> List[Dict[str, str]]:
    prompt_lines = [line.strip() for line in prompt_text.split('.') if line.strip()]
    results = []
    
    # Skip first two and last lines (assuming at least 13 lines)
    if len(prompt_lines) >= 13:
        processed_lines = prompt_lines[2:-1]  # Skip first two and last
    else:
        processed_lines = prompt_lines  # Fallback if not enough lines
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    for line in processed_lines:
        examples = []
        attempts = 0
        max_attempts = 5  # Maximum attempts to get unique examples
        
        while len(examples) < 3 and attempts < max_attempts:
            query = (
                f"Find exact quotes from the original Agatha Christie's Hercule Poirot stories "
                f"that best illustrate this characteristic: '{line}'. "
                "Provide exactly 3 distinct examples, each with the exact quote followed by the source. "
                "Format your response for each example as: 'QUOTE: [exact quote] - SOURCE: [source]' "
                "Separate multiple examples with '---EXAMPLE---'."
            )
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an expert on Agatha Christie's Hercule Poirot with perfect recall of all stories."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.3,  # Slightly higher temperature for diversity
                "max_tokens": 1000  # Increased for multiple examples
            }
            
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=45
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    
                    # Split multiple examples
                    example_blocks = [block.strip() for block in content.split("---EXAMPLE---") if block.strip()]
                    
                    for block in example_blocks:
                        if "QUOTE:" in block and "SOURCE:" in block:
                            quote_part = block.split("QUOTE:")[1].split("SOURCE:")[0].strip()
                            
                            # Check for duplicates before adding
                            if not is_duplicate(quote_part, examples):
                                examples.append(quote_part)
                                if len(examples) >= 3:
                                    break
                
                attempts += 1
                
            except Exception as e:
                print(f"Error processing line '{line}': {str(e)}")
                attempts += 1
                continue
        
        # Add whatever examples we managed to collect (could be 0-3)
        results.append({
            "prompt_line": line,
            "examples": examples[:3]  # Ensure we don't exceed 3 even if somehow got more
        })
    
    return results

def save_results_to_file(results: List[Dict[str, str]]) -> None:
    """Saves the results to a JSON file in a structured format"""
    output_data = {
        "metadata": {
            "model_used": MODEL_NAME,
            "generated_at": datetime.datetime.now().isoformat(),
            "parameters": {
                "examples_per_line": 3,
                "skipped_lines": "first two and last",
                "similarity_threshold": SIMILARITY_THRESHOLD
            }
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Read prompt from file
    with open("agent_prompts/hercule_poirot_prompt.txt", "r", encoding="utf-8") as f:
        prompt_content = f.read()
    
    # Get matches
    matched_examples = find_examples(prompt_content)
    
    # Save results to file
    save_results_to_file(matched_examples)