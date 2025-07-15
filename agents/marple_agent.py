


from llm_config import get_llama_llm
from agents.base_agent import DetectiveAgent
import os

def create_marple_agent():
    llm = get_llama_llm()
    return DetectiveAgent("Marple", "prompts/marple_prompt.yaml", llm)
