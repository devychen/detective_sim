import os
from abc import ABC, abstractmethod
from typing import Dict, Any
import yaml
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

class DetectiveAgent(ABC):
    def __init__(self, prompt_path: str):
        self.llm = self._initialize_llm()
        self.prompt = self._load_prompt(prompt_path)
        self.memory = []
        
    def _initialize_llm(self):
        return ChatNVIDIA(
            model=os.getenv("MODEL_NAME", "meta/llama-3-70b-instruct"),
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9
        )
    
    def _load_prompt(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def format_prompt(self, case_info: str) -> str:
        character_desc = "\n".join([item['description'] for item in self.prompt if 'description' in item])
        examples = "\n".join([item['example'] for item in self.prompt if 'example' in item])
        
        return f"""
        You are to strictly follow these character descriptions:
        {character_desc}
        
        Here are examples of how you should speak:
        {examples}
        
        Current case information:
        {case_info}
        
        Remember to stay in character at all times.
        """
    
    @abstractmethod
    def generate_response(self, case_info: str, discussion_history: list = None) -> str:
        pass
    
    def add_to_memory(self, message: str, is_user: bool = False):
        if is_user:
            self.memory.append(HumanMessage(content=message))
        else:
            self.memory.append(AIMessage(content=message))