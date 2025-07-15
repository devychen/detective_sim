
# holmes_agent.py

from llm_config import get_llama_llm
from agents.base_agent import DetectiveAgent
import os

def create_holmes_agent():
    llm = get_llama_llm()
    return DetectiveAgent("Holmes", "prompts/holmes_prompt.yaml", llm)
