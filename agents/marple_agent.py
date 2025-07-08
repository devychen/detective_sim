
from agents.base_agent import DetectiveAgent

class PoirotAgent(DetectiveAgent):
    def __init__(self):
        super().__init__(name="Poirot", prompt_path="prompts/marple_prompt.yaml")

print(" MarpleAgent is defined.")