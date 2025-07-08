# agents/marple_agent.py

from agents.base_agent import DetectiveAgent

class MarpleAgent(DetectiveAgent):
    def __init__(self):
        super().__init__(name="Marple", prompt_path="prompts/poirot_prompt.yaml")

print(" PoirotAgent is defined.")