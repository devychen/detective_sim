# pip install python-dotenv

import yaml
import openai
import os
from dotenv import load_dotenv


# Set OpenAI API key
load_dotenv('key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


# 读取分析文件内容
with open("profiles/sherlock_holmes_analysis.txt", "r") as file:
    sherlock_analysis = file.read()

# 构建给GPT的指令
system_message = """You are a professional literacy analysis assistant. Please process the text strictly according to these requirements：
1. Extract the key traits that identify the character in the following text and produce a single, cohesive prompt suitable for LLM character simulation or prompt engineering. 
2. The output must consist of exactly ten sentences, with two sentences each dedicated to the character’s vocabulary, sentence structure, discourse pattern, personality traits, and investigative methods.
3. Do not include any bullet points, lists, formatting, or section headings.
4. The content of each sentence must be based strictly on the original text; do not invent or infer beyond what is explicitly stated.
5. Begin the prompt with: "You are {name}. You are {occupation}."
6. End the prompt with: "Stay in character at all times." """

# 调用OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": sherlock_analysis}
    ],
    temperature=0.1  # 降低随机性确保准确性
)

# 获取生成的prompt
generated_prompt = response.choices[0].message.content

# 保存到文件
with open("sh_prompt", "w") as file:
    file.write(generated_prompt)

print("Prompt successfully generated and saved to sh_prompt file")