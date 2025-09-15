# llm_config.py
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# 读取 NVIDIA Key
load_dotenv("nvidia_key_8b.env")

def get_llama_llm(model_name="meta/llama3-8b-instruct") -> ChatNVIDIA:
    return ChatNVIDIA(
        model=model_name,
        temperature=0.5,   #
        max_tokens=1024,   
        request_timeout=60,
        top_p=0.9,
        n=1,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
