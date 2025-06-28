# pip install openai python-dotenv

import openai
import yaml
import os
from dotenv import load_dotenv

# 初始化
load_dotenv('openai_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

# 确保密钥已正确加载
if not openai.api_key:
    raise ValueError("API key not found, please check the file key.env")

def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def generate_content(detective_name, template):
    results = {}
    for category, data in template['categories'].items():
        task = data['task'].replace('D_name', detective_name)
        requirements = '\n'.join(data['requirements']).replace('D_name', detective_name)
        
        prompt = f"{task}\n\nRequirements:\n{requirements}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a linguistic analysis expert specializing in fictional detective characters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        results[category] = response.choices[0].message['content'].strip()
    
    return results

def save_results(detective_name, results):
    file_name = f"{detective_name.replace(' ', '_').lower()}_analysis.txt"
    with open(file_name, 'w', encoding='utf-8') as file:
        for category, content in results.items():
            file.write(f"=== {category.upper()} ===\n")
            file.write(content + '\n\n')

def main():
    template = load_template('prep_prompts/profile_prompt.yaml')
    detectives = ['Sherlock Holmes', 'Hercule Poirot', 'Miss Marple']
    
    for detective in detectives:
        print(f"Generating content for {detective}...")
        results = generate_content(detective, template)
        save_results(detective, results)
        print(f"Content for {detective} saved successfully.")

if __name__ == "__main__":
    main()