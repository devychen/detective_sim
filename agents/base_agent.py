# agents/base_agent.py

import yaml
import dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.nvidia_llm import NvidiaLLM

# Load API key from .env
dotenv.load_dotenv("nvidia_key.env")
API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"

class DetectiveAgent:
    def __init__(self, name: str, prompt_path: str, temperature=0.7):
        self.name = name
        self.prompt = self._load_prompt(prompt_path)
        self.llm = NvidiaLLM(
            model_name=MODEL_NAME,
            api_url=API_URL,
            api_key=API_KEY,
            temperature=temperature
        )
        self.chain = self._build_chain()

    def _load_prompt(self, path: str) -> PromptTemplate:
        with open(path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        return PromptTemplate(
            input_variables=content.get("input_variables", ["input"]),
            template=content["template"]
        )

    def _build_chain(self) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    def run(self, input_text: str) -> str:
        response = self.chain.run({"input": input_text})
        return response
