# agents/holmes_agent.py

from agents.base_agent import DetectiveAgent
from llm_config import get_llama_llm
import os

PROMPT_PATH = os.path.join("prompts", "marple_prompt.yaml")

def create_marple_agent():
    llm = get_llama_llm()
    return DetectiveAgent(name="Miss Marple", prompt_path=PROMPT_PATH, llm=llm)


print(" MarpleAgent is defined.")
