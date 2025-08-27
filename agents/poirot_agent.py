
# agents/poirot_agent.py

from .base_agent import BaseAgent

class PoirotAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(name="Hercule Poirot", yaml_path="prompts/poirot.yaml", llm=llm)
