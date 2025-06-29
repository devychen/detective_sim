import os
import requests
import json
import dotenv
from typing import Dict, List, Tuple
import datetime  # Moved to top level

# Load environment variables
dotenv.load_dotenv("nvidia_key.env")  # Ensure your API key is in this file

# API Configuration
MODEL_NAME = "meta/llama-3.3-70b-instruct"
API_URL = "https://integrate.api.nvidia.com/v1"
OUTPUT_FILE = "sh_examples.json"

def find_holmes_examples(prompt_text: str) -> List[Dict[str, str]]:
    """
    Uses NVIDIA API to find examples from Sherlock Holmes canon that match each line in the prompt.
    
    Args:
        prompt_text: The full Sherlock Holmes character prompt text
        
    Returns:
        List of dictionaries containing prompt_line and a list of examples
    """
    prompt_lines = [line.strip() for line in prompt_text.split('.') if line.strip()]
    results = []
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    for line in prompt_lines:
        query = (
            f"Find three distinct exact quotes from the original Sherlock Holmes stories by Arthur Conan Doyle "
            f"that best illustrate this characteristic: '{line}'. "
            "For each quote, provide the exact text followed by the source (story title and chapter). "
            "Format your response exactly like this for each example:\n"
            "1. QUOTE: [exact quote] - SOURCE: [source]\n"
            "2. QUOTE: [exact quote] - SOURCE: [source]\n"
            "3. QUOTE: [exact quote] - SOURCE: [source]"
        )
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a Sherlock Holmes expert with perfect recall of all canonical stories."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 1000  # Increased to accommodate more examples
        }
        
        try:
            response = requests.post(
                f"{API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=45  # Increased timeout
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                examples = []
                
                # Parse multiple examples from response
                for i in range(1, 4):
                    prefix = f"{i}. QUOTE:"
                    if prefix in content:
                        quote_part = content.split(prefix)[1].split("SOURCE:")[0].strip()
                        examples.append(quote_part)
                
                if examples:
                    results.append({
                        "prompt_line": line, 
                        "examples": examples,
                        "raw_response": content if len(examples) < 3 else None  # Store raw if we didn't get 3
                    })
                else:
                    results.append({
                        "prompt_line": line, 
                        "examples": [f"No properly formatted examples found. Raw response: {content}"]
                    })
            else:
                results.append({
                    "prompt_line": line, 
                    "examples": [f"API request failed with status {response.status_code}"]
                })
                
        except Exception as e:
            results.append({
                "prompt_line": line, 
                "examples": [f"Error occurred: {str(e)}"]
            })
    
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