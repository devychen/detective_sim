# base_agent.py

import yaml
from pathlib import Path
from llm_config import get_llama_llm

class BaseAgent:
    def __init__(self, name: str, yaml_path: str, llm=None):
        self.name = name
        self.yaml_path = Path(yaml_path)
        self.llm = llm or get_llama_llm()
        self.memory = []  # 存储 (speaker, content)
        self.config = self._load_config()

    def _load_config(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def update_memory(self, speaker: str, content: str):
        """把对话内容存到 memory 里"""
        self.memory.append({"speaker": speaker, "content": content})

    def build_prompt(self, task_instruction: str = "") -> str:
        """拼接 role_play + protective + task + memory"""
        parts = []

        # 角色扮演部分
        if "role_play" in self.config:
            for item in self.config["role_play"]:
                parts.append(item.get("description", ""))

        # protective 限制
        if "protective" in self.config:
            parts.append("Protective rules:")
            for p in self.config["protective"]:
                parts.append(f"- {p['description']}")

        # memory（历史对话）
        if self.memory:
            history_str = "\n".join([f"{m['speaker']}: {m['content']}" for m in self.memory])
            parts.append("Conversation so far:\n" + history_str)

        # 任务
        if "task" in self.config:
            parts.append("Task:\n" + self.config["task"])

        if task_instruction:
            parts.append(task_instruction)

        return "\n".join(parts)

    def run(self, task_instruction: str = "") -> str:
        """执行一步推理"""
        prompt = self.build_prompt(task_instruction)
        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        self.update_memory(self.name, text)
        return text
