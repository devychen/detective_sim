# agents/holmes_agent.py
from agents.base_agent import BaseAgent
from llm_config import get_llama_llm

def create_agent():
    return BaseAgent(name="Hercule Poirot", llm=get_llama_llm())
