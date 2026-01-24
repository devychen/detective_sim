# llm_config.py
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# 读取 NVIDIA Key


load_dotenv("nvidia_key_3b.env")

def get_llama_llm(model_name="meta/llama-3.2-3b-instruct") -> ChatNVIDIA:
    return ChatNVIDIA(
        model=model_name,
        temperature=0.7,   
        max_tokens=512,  # 256, 512, 1024  
        request_timeout=60,
        top_p=0.9,
        n=1,
        api_key=os.getenv("NVIDIA_API_KEY")
    )