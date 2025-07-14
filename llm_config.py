# llm_config.py
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv("nvidia_key.env")

def get_llama_llm(model_name="meta/llama-3.3-70b-instruct") -> ChatNVIDIA:
    return ChatNVIDIA(
        model=model_name,
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        n=1,
        api_key=os.getenv("NVIDIA_API_KEY")
    )
