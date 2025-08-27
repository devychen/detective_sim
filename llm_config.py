# llm_config.py
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# 读取 NVIDIA Key
load_dotenv("nvidia_key.env")

def get_llama_llm(model_name="meta/llama-3.3-70b-instruct") -> ChatNVIDIA:
    return ChatNVIDIA(
        model=model_name,
        temperature=0.5,   # 稍微保守
        max_tokens=512,   # 可以调大点，比如 1024
        request_timeout=60,
        top_p=0.9,
        n=1,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
