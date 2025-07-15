# agents/holmes_agent.py

from agents.base_agent import DetectiveAgent
from llm_config import get_llama_llm
import os

PROMPT_PATH = os.path.join("prompts", "poirot_prompt.yaml")

def create_poirot_agent():
    llm = get_llama_llm()
    return DetectiveAgent(name="Hercule Poirot", prompt_path=PROMPT_PATH, llm=llm)


print(" PoirotAgent is defined.")
