# agents/holmes_agent.py
from .base_agent import BaseAgent

class HolmesAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(name="Holmes", yaml_path="prompts/holmes.yaml", llm=llm)


