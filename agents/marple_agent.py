from .base_agent import DetectiveAgent
from typing import Optional, List

class MissMarpleAgent(DetectiveAgent):
    def __init__(self):
        super().__init__("config/prompts/marple_prompt.yaml")
    
    def generate_response(self, case_info: str, discussion_history: Optional[List[str]] = None) -> str:
        prompt = self.format_prompt(case_info)
        
        if discussion_history:
            history_text = "\n".join([f"{msg['speaker']} said: {msg['content']}" 
                                    for msg in discussion_history])
            prompt += f"\n\nMy dears, here's what we've discussed so far:\n{history_text}\n\nWhat would Miss Marple say next?"
        
        response = self.llm.invoke(prompt)
        self.add_to_memory(response.content)
        return response.content