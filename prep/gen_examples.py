import os
import requests
import json
import dotenv
import difflib
from typing import Dict, List, Tuple, Optional
import datetime

# Load environment variables
dotenv.load_dotenv("nvidia_key.env")

# API Configuration
MODEL_NAME = "meta/llama-3.3-70b-instruct"
API_URL = "https://integrate.api.nvidia.com/v1"
OUTPUT_FILE = "sh_examples.json"
SIMILARITY_THRESHOLD = 0.9  # 90% similarity is considered duplicate
MAX_ATTEMPTS = 3  # Max attempts to get unique examples

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate the similarity ratio between two texts using difflib"""
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def is_too_similar(new_example: str, existing_examples: List[str]) -> bool:
    """Check if new example is too similar to any existing examples"""
    for existing in existing_examples:
        if calculate_similarity(new_example, existing) > SIMILARITY_THRESHOLD:
            return True
    return False

def parse_examples(content: str) -> List[str]:
    """Parse examples from API response content"""
    examples = []
    for i in range(1, 4):
        prefix = f"{i}. QUOTE:"
        if prefix in content:
            parts = content.split(prefix)[1].split("SOURCE:") 
            if len(parts) > 1:
                quote_part = parts[0].strip()
                examples.append(quote_part)
    return examples

def find_holmes_examples(prompt_text: str) -> List[Dict[str, str]]:
    """
    Uses NVIDIA API to find examples from Sherlock Holmes canon that match each line in the prompt,
    ensuring all examples have low similarity.
    """
    prompt_lines = [line.strip() for line in prompt_text.split('.') if line.strip()]
    results = []
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    for line in prompt_lines:
        examples = []
        attempts = 0
        raw_responses = []
        
        while len(examples) < 3 and attempts < MAX_ATTEMPTS:
            query = (
                f"Find distinct exact quotes from the original Sherlock Holmes stories by Arthur Conan Doyle "
                f"that best illustrate this characteristic: '{line}'. "
                "Provide three completely different examples that demonstrate different aspects of this trait. "
                "For each quote, provide the exact text followed by the source (story title and chapter). "
                "Format your response exactly like this for each example:\n"
                "1. QUOTE: [exact quote] - SOURCE: [source]\n"
                "2. QUOTE: [exact quote] - SOURCE: [source]\n"
                "3. QUOTE: [exact quote] - SOURCE: [source]"
            )
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a Sherlock Holmes expert with perfect recall of all canonical stories. Provide diverse examples that show different facets of each characteristic."},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.3,  # Slightly increased for diversity
                "max_tokens": 1000
            }
            
            try:
                response = requests.post(
                    f"{API_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    raw_responses.append(content)
                    new_examples = parse_examples(content)
                    
                    for example in new_examples:
                        if not is_too_similar(example, examples) and example not in examples:
                            examples.append(example)
                            if len(examples) >= 3:
                                break
                
                attempts += 1
                
            except Exception as e:
                results.append({
                    "prompt_line": line, 
                    "examples": [f"Error occurred: {str(e)}"],
                    "attempts": attempts + 1
                })
                continue
        
        if examples:
            results.append({
                "prompt_line": line,
                "examples": examples[:3],  # Ensure max 3 examples
                "attempts": attempts,
                "unique_examples": len(examples),
                "raw_responses": raw_responses if len(examples) < 3 else None
            })
        else:
            results.append({
                "prompt_line": line,
                "examples": ["Failed to get sufficient unique examples after multiple attempts"],
                "attempts": attempts,
                "raw_responses": raw_responses
            })
    
    return results

def save_results_to_file(results: List[Dict[str, str]]) -> None:
    """Saves the results to a JSON file in a structured format"""
    output_data = {
        "metadata": {
            "model_used": MODEL_NAME,
            "generated_at": datetime.datetime.now().isoformat(),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "max_attempts": MAX_ATTEMPTS
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Read prompt from file
    with open("agent_prompts/sherlock_holmes_prompt.txt", "r", encoding="utf-8") as f:
        prompt_content = f.read()
    
    # Get matches
    matched_examples = find_holmes_examples(prompt_content)
    
    # Save results to file
    save_results_to_file(matched_examples)