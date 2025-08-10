# base_agent.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import yaml
import time

class DetectiveAgent:
    def __init__(self, name: str, prompt_path: str, llm: Runnable):
        self.name = name
        self.llm = llm
        self.prompt = self._build_prompt(prompt_path)
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=False)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def _build_prompt(self, path: str) -> PromptTemplate:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        descriptions = []
        for item in data:
            if 'description' in item:
                descriptions.append(item['description'])
            if 'instruction' in item:
                descriptions.append(item['instruction'])

        # Add chat_history placeholder in the prompt template
        prompt_text = "\n".join(descriptions) + "\n\nCurrent Chat History:\n{chat_history}\n\n{input}"
        return PromptTemplate.from_template(prompt_text)

    def update_memory(self, speaker: str, text: str):
        """Update memory to record other agents' speech"""
        if speaker != self.name:  # Avoid duplicate recording of own speech
            self.memory.save_context({"input": f"{speaker} said: {text}"}, {"output": ""})

    def run(self, input_text: str) -> str:
        for attempt in range(3):
            try:
                response = self.chain.invoke({"input": input_text})
                return response['text']
            except Exception as e:
                print(f"[{self.name}] Retry {attempt+1}/3 due to error: {e}")
                time.sleep(5)
        raise RuntimeError(f"{self.name} failed after 3 retries.")
