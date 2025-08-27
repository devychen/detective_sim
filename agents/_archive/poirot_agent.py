

from llm_config import get_llama_llm
from agent._archive.base_agent import DetectiveAgent
import os

def create_poirot_agent():
    llm = get_llama_llm()
    return DetectiveAgent("Poirot", "prompts/poirot_prompt.yaml", llm)
