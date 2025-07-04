from detective_sim.agents.base_agent import DetectiveAgent
from typing import Optional, List

class HerculePoirotAgent(DetectiveAgent):
    def __init__(self):
        super().__init__("config/prompts/poirot_prompt.yaml")
    
    def generate_response(self, case_info: str, discussion_history: Optional[List[str]] = None) -> str:
        prompt = self.format_prompt(case_info)
        
        if discussion_history:
            history_text = "\n".join([f"Detective {msg['speaker']}: {msg['msg']}" 
                                    for msg in discussion_history])
            prompt += f"\n\nCurrent discussion:\n{history_text}\n\nWhat does Hercule Poirot say?"
        
        response = self.llm.invoke(prompt)
        self.add_to_memory(response.content)
        return response.content