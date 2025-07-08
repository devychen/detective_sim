# agents/holmes_agent.py

from agents.base_agent import DetectiveAgent

class HolmesAgent(DetectiveAgent):
    def __init__(self):
        super().__init__(name="Holmes", prompt_path="prompts/holmes_prompt.yaml")

print(" HolmesAgent is defined.")
