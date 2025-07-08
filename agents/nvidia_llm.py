# agents/nvidia_llm.py

import os
import requests
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from typing import List

class NvidiaLLM(LLM):
    def __init__(self, model_name: str, api_url: str, api_key: str, temperature: float = 0.7):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        response = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "nvidia-chat-llm"
