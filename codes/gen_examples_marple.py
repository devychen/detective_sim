import os
import requests
import json
import dotenv
import datetime
from typing import Dict, List

# Load environment variables
dotenv.load_dotenv("nvidia_key.env")

# Configuration
MODEL_NAME = "meta/llama-3.3-70b-instruct"
API_URL = "https://integrate.api.nvidia.com/v1"
OUTPUT_FILE = "marple_examples.json"

def find_marple_examples(prompt_text: str) -> List[Dict[str, str]]:
    prompt_lines = [line.strip() for line in prompt_text.split('.') if line.strip()]
    results = []
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    for line in prompt_lines:
        query = (
            f"Find one exact quote from Agatha Christie's Miss Marple stories "
            f"that best illustrates this characteristic: '{line}'. "
            "Provide the exact quote followed by the source (story title). "
            "Format: 'QUOTE: [exact quote] - SOURCE: [source]'"
        )
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an expert on Agatha Christie's Miss Marple with perfect recall of all stories."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }
        
        try:
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
                    results.append({"prompt_line": line, "example": quote_part})
                else:
                    results.append({"prompt_line": line, "example": f"No properly formatted example found. Raw response: {content}"})
            else:
                results.append({"prompt_line": line, "example": f"API request failed with status {response.status_code}"})
                
        except Exception as e:
            results.append({"prompt_line": line, "example": f"Error occurred: {str(e)}"})
    
    return results

def save_results_to_file(results: List[Dict[str, str]]) -> None:
    output_data = {
        "metadata": {
            "model_used": MODEL_NAME,
            "generated_at": datetime.datetime.now().isoformat(),
            "character": "Miss Marple",
            "author": "Agatha Christie"
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    with open("agent_prompts/miss_marple_prompt.txt", "r", encoding="utf-8") as f:
        prompt_content = f.read()
    
    matched_examples = find_marple_examples(prompt_content)
    save_results_to_file(matched_examples)