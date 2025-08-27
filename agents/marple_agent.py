
# agents/marple_agent.py

from .base_agent import BaseAgent

class MarpleAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(name="Miss Marple", yaml_path="prompts/marple.yaml", llm=llm)
