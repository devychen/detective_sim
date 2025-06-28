# pip install python-dotenv

import yaml
import openai
import os
from dotenv import load_dotenv


# Initialize OpenAI client
load_dotenv('key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_prompt_template():
    with open('prompts/character_prompt.yaml', 'r') as file:
        return yaml.safe_load(file)

def parse_analysis_file(file_path):
    sections = {
        'VOCABULARY': '',
        'SENTENCES': '',
        'DISCOURSE_PATTERN': '',
        'PERSONAL_TRAITS': ''
    }
    current_section = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('===') and line.endswith('==='):
                section_name = line.strip('= ').upper()
                if section_name in sections:
                    current_section = section_name
                else:
                    current_section = None
            elif current_section and line:
                sections[current_section] += line + ' '
    
    return sections

def generate_character_prompt(template, analysis, name, occupation):
    # 提取每个部分的前两句话（不处理句式，直接按原始文本填充）
    def extract_sentences(text, n=2):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[:n] if len(sentences) >= n else sentences + [""] * (n - len(sentences))
    
    vocab = extract_sentences(analysis['VOCABULARY'])
    sentences = extract_sentences(analysis['SENTENCES'])
    discourse = extract_sentences(analysis['DISCOURSE_PATTERN'])
    personality = extract_sentences(analysis['PERSONAL_TRAITS'])
    
    # 用 YAML 模板填充变量（假设模板是字符串形式）
    prompt = template['character_prompt']['template'].format(
        name=name,
        occupation=occupation,
        vocabulary_1=vocab[0],
        vocabulary_2=vocab[1],
        sentence_1=sentences[0],
        sentence_2=sentences[1],
        discourse_1=discourse[0],
        discourse_2=discourse[1],
        personality_1=personality[0],
        personality_2=personality[1],
        example_1="[quote from works]",  # 替换为实际引用
        example_2="[quote from works]",
        example_3="[quote from works]",
        example_4="[quote from works]",
        example_5="[quote from works]",
        example_6="[quote from works]",
        example_7="[quote from works]",
        example_8="[quote from works]",
    )
    return prompt

def save_prompt_to_file(character_name, prompt):
    filename = f"{character_name.lower().replace(' ', '_')}_prompt.txt"
    with open(filename, 'w') as file:
        file.write(prompt)
    print(f"Saved prompt for {character_name} to {filename}")

def main():
    # Load the prompt template
    template = load_prompt_template()
    
    # Process each detective
    detectives = [
        {'name': 'Sherlock Holmes', 'occupation': 'a consulting detective', 'analysis_file': 'profiles/sherlock_holmes_analysis.txt'},
        {'name': 'Hercule Poirot', 'occupation': 'a private detective', 'analysis_file': 'profiles/hercule_poirot_analysis.txt'},
        {'name': 'Miss Marple', 'occupation': 'an amateur detective', 'analysis_file': 'profiles/miss_marple_analysis.txt'}
    ]
    
    for detective in detectives:
        try:
            analysis = parse_analysis_file(detective['analysis_file'])
            prompt = generate_character_prompt(template, analysis, detective['name'], detective['occupation'])
            save_prompt_to_file(detective['name'], prompt)
        except FileNotFoundError:
            print(f"Analysis file not found for {detective['name']}: {detective['analysis_file']}")
        except Exception as e:
            print(f"Error processing {detective['name']}: {str(e)}")

if __name__ == "__main__":
    main()