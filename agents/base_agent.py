# base_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import yaml

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

        # prompt 模板里加 chat_history 占位符
        prompt_text = "\n".join(descriptions) + "\n\nCurrent Chat History：\n{chat_history}\n\n{input}"
        return PromptTemplate.from_template(prompt_text)

    def run(self, input_text: str) -> str:
        response = self.chain.invoke({"input": input_text})
        return response['text']
