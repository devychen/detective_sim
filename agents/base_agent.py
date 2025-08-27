# base_agent.py

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema.runnable import Runnable

class DetectiveAgent:
    def __init__(self, name: str, llm: Runnable):
        self.name = name
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = LLMChain(llm=self.llm, memory=self.memory)

    def update_memory(self, role: str, content: str):
        """手动把别的 agent 的发言放进记忆"""
        self.memory.chat_memory.add_message({"role": role, "content": content})

    def run(self, input_text: str) -> str:
        """调用 LLMChain"""
        response = self.chain.invoke({
            "input": input_text,
            "chat_history": self.memory.load_memory_variables({})["chat_history"]
        })
        return response["text"]
