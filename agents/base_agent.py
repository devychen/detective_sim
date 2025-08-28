# agents/base_agent.py
from typing import List, Dict, Any

class BaseAgent:
    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm
        self.memory: List[Dict[str, str]] = []  # conversation history

    def run(self, prompt: str) -> str:
        """调用 LLM 并返回结果"""
        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()

    def update_memory(self, role: str, content: str):
        """更新 memory，只存 agent 之间的对话"""
        self.memory.append({"role": role, "content": content})

    def get_memory_text(self) -> str:
        """把 memory 格式化成 conversation so far"""
        if not self.memory:
            return "No prior conversation."
        formatted = "\n".join([f"{m['role']}: {m['content']}" for m in self.memory])
        return f"=== Conversation So Far ===\n{formatted}\n"
