import openai
import os
from dotenv import load_dotenv


# Set OpenAI API key
load_dotenv('openai_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the characters and their files
character_files = {
    "Sherlock Holmes": "profiles/sherlock_holmes_analysis.txt",
    "Hercule Poirot": "profiles/hercule_poirot_analysis.txt",
    "Miss Marple": "profiles/miss_marple_analysis.txt"
}

# System message template for GPT
system_message = """You are a professional literary analysis assistant. Please process the text strictly according to these requirements:
1. Extract the character's key traits
2. Generate an 8-sentence character prompt
3. First two sentences describe vocabulary traits, next two describe sentence structure, then two for discourse pattern, and finally two for personality
4. Begin with "You are {name}. You are {occupation}."
5. End with "Stay in character at all times."
6. Do not use any bullet points, lists, or formatting
7. Use only content explicitly stated in the original text"""

def generate_prompt(character_name, analysis_file):
    """Generate a character prompt from an analysis file"""
    try:
        with open(analysis_file, "r") as file:
            character_analysis = file.read()
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": character_analysis}
            ],
            temperature=0.1
        )
        
        generated_prompt = response.choices[0].message.content
        
        # Save to file
        output_filename = f"{character_name.lower().replace(' ', '_')}_prompt"
        with open(output_filename, "w") as file:
            file.write(generated_prompt)
        
        print(f"Successfully generated {output_filename}")
        return True
    
    except Exception as e:
        print(f"Error processing {character_name}: {str(e)}")
        return False

# Process all characters
for character, analysis_file in character_files.items():
    generate_prompt(character, analysis_file)

print("All character prompts generated successfully!")