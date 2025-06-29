import os
import requests
import json
import dotenv
from typing import Dict, List, Tuple
import datetime
from collections import Counter
import math

# Load environment variables
dotenv.load_dotenv("nvidia_key.env")  # Ensure your API key is in this file

# API Configuration
MODEL_NAME = "meta/llama-3.3-70b-instruct"
API_URL = "https://integrate.api.nvidia.com/v1"
OUTPUT_FILE = "sh_examples.json"

def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Calculate the percentage of overlapping words between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Percentage of overlapping words (0-100)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    smaller_set = min(len(words1), len(words2))
    return (len(intersection) / smaller_set) * 100

def is_too_similar(new_example: str, existing_examples: List[str], threshold: float = 20.0) -> bool:
    """
    Check if a new example is too similar to any existing examples.
    
    Args:
        new_example: The new example text to check
        existing_examples: List of already accepted examples
        threshold: Maximum allowed word overlap percentage
        
    Returns:
        True if too similar to any existing example, False otherwise
    """
    for example in existing_examples:
        overlap = calculate_word_overlap(new_example, example)
        if overlap > threshold:
            return True
    return False

def find_holmes_examples(prompt_text: str) -> List[Tuple[str, str]]:
    """
    Uses NVIDIA API to find examples from Sherlock Holmes canon that match each line in the prompt.
    
    Args:
        prompt_text: The full Sherlock Holmes character prompt text
        
    Returns:
        List of tuples containing (prompt_line, example_from_text)
    """
    prompt_lines = [line.strip() for line in prompt_text.split('.') if line.strip()]
    results = []
    collected_examples = []
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    for line in prompt_lines:
        query = (
            f"Find one exact quote from the original Sherlock Holmes stories by Arthur Conan Doyle "
            f"that best illustrates this characteristic: '{line}'. "
            "Provide the exact quote followed by the source (story title and chapter). "
            "The quote should be distinct from previous examples with no more than 40% word overlap. "
            "Format your response: 'QUOTE: [exact quote] - SOURCE: [source]'"
        )
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a Sherlock Holmes expert with perfect recall of all canonical stories. Provide distinct examples with minimal word overlap."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.3,  # Slightly increased for variety
            "max_tokens": 500
        }
        
        try:
            max_attempts = 3
            attempts = 0
            success = False
            
            while attempts < max_attempts and not success:
                response = requests.post(
                    f"{API_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    if "QUOTE:" in content and "SOURCE:" in content:
                        quote_part = content.split("QUOTE:")[1].split("SOURCE:")[0].strip()
                        
                        # Check for similarity with existing examples
                        if not is_too_similar(quote_part, collected_examples):
                            collected_examples.append(quote_part)
                            results.append({"prompt_line": line, "example": quote_part})
                            success = True
                        else:
                            attempts += 1
                            if attempts < max_attempts:
                                print(f"Example too similar to previous ones, retrying... (attempt {attempts})")
                    else:
                        results.append({"prompt_line": line, "example": f"No properly formatted example found. Raw response: {content}"})
                        success = True  # Give up on this line
                else:
                    results.append({"prompt_line": line, "example": f"API request failed with status {response.status_code}"})
                    success = True  # Give up on this line
                
            if attempts >= max_attempts and not success:
                results.append({"prompt_line": line, "example": "Failed to find sufficiently unique example after multiple attempts"})
                
        except Exception as e:
            results.append({"prompt_line": line, "example": f"Error occurred: {str(e)}"})
    
    return results

def save_results_to_file(results: List[Dict[str, str]]) -> None:
    """Saves the results to a JSON file in a structured format"""
    output_data = {
        "metadata": {
            "model_used": MODEL_NAME,
            "generated_at": datetime.datetime.now().isoformat()
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