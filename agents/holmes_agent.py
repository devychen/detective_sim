from .base_agent import DetectiveAgent
from typing import Optional, List

class SherlockHolmesAgent(DetectiveAgent):
    def __init__(self):
        super().__init__("config/prompts/holmes_prompt.yaml")
    
    def generate_response(self, case_info: str, discussion_history: Optional[List[str]] = None) -> str:
        prompt = self.format_prompt(case_info)
        
        if discussion_history:
            history_text = "\n".join([f"Detective {msg['speaker']}: {msg['content']}" 
                                    for msg in discussion_history])
            prompt += f"\n\nDiscussion so far:\n{history_text}\n\nYour response:"
        
        response = self.llm.invoke(prompt)
        self.add_to_memory(response.content)
        return response.content