import openai
import os
import yaml
from dotenv import load_dotenv

# Set OpenAI API key
load_dotenv('openai_key.env')
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the characters and their files (same as original)
character_files = {
    "Sherlock Holmes": "profiles/sherlock_holmes_analysis.txt",
    "Hercule Poirot": "profiles/hercule_poirot_analysis.txt",
    "Miss Marple": "profiles/miss_marple_analysis.txt"
}

# System message template for protective prompts
system_message = """You are a professional literary analysis assistant specialized in character behavior boundaries. 
Generate protective prompts that prevent the character from behaving out-of-character. Follow these requirements:
1. Analyze the character's core traits and identify behaviors they would NEVER do
2. Generate 5-7 protective prompts in YAML list format
3. Each prompt should begin with "You are not supposed to..."
4. Focus on actions, speech patterns, attitudes that contradict the character's nature
5. Include both general behavior restrictions and specific dialogue patterns to avoid
6. Base all prompts strictly on the character's established traits
7. Do not include any positive behaviors - only what to avoid
8. Keep each description concise (one sentence)
9. Maintain YAML syntax with proper indentation

Example format:
- description: You are not supposed to engage in small talk.
- description: You are not supposed to show emotional vulnerability.
- description: You are not supposed to speak in informal slang."""

def generate_protective_prompts(character_name, analysis_file):
    """Generate protective prompts from character analysis"""
    try:
        with open(analysis_file, "r") as file:
            character_analysis = file.read()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": character_analysis}
            ],
            temperature=0.2  # Slightly higher for creative negatives
        )
        
        generated_prompts = response.choices[0].message.content
        
        # Save to YAML file
        output_filename = f"protective_{character_name.lower().replace(' ', '_')}.yaml"
        with open(output_filename, "w") as file:
            file.write(generated_prompts)
        
        print(f"Successfully generated {output_filename}")
        return True
    
    except Exception as e:
        print(f"Error processing {character_name}: {str(e)}")
        return False

# Process all characters
for character, analysis_file in character_files.items():
    generate_protective_prompts(character, analysis_file)

print("All protective prompts generated successfully!")