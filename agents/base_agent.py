# agents/base_agent.py
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from llm_config import get_llama_llm

class BaseAgent:
    """通用探员：加载自身 YAML 配置；维护 memory；构造完整 Prompt；调用 LLM。"""
    def __init__(self, name: str, yaml_path: str, llm=None):
        self.name = name
        self.yaml_path = Path(yaml_path)
        self.llm = llm or get_llama_llm()
        self.memory: List[Dict[str, str]] = []   # [{speaker, content}, ...]
        self.config: Dict[str, Any] = self._load_config()
        self.context: Dict[str, Any] = {}        # 案件上下文（setting/victim/...）

    # -------- public API --------
    def set_context(self, ctx: Dict[str, Any]) -> None:
        """由协作管理器注入案件上下文，供 task.format(**ctx) 使用"""
        self.context = ctx or {}

    def update_memory(self, speaker: str, content: str) -> None:
        """把某条消息加入记忆（系统、他人、自己都行）"""
        self.memory.append({"speaker": speaker, "content": content})

    def run(self, task_instruction: str = "") -> Tuple[str, str]:
        """
        执行一步推理。
        返回: (模型回复文本, 实际发送的完整 Prompt)，方便外部落盘复盘。
        """
        prompt = self.build_prompt(task_instruction)
        raw = self.llm.invoke(prompt)
        text = self._extract_text(raw)
        # 把自己的输出也写入记忆
        self.update_memory(self.name, text)
        return text, prompt

    # -------- internals --------
    def _load_config(self) -> Dict[str, Any]:
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def build_prompt(self, task_instruction: str = "") -> str:
        """
        最终 Prompt = 角色风格 + Protective + 任务(带上下文格式化) + 历史对话 + 本轮指令
        """
        parts: List[str] = []

        # 角色扮演风格（仅文本+可选示例）
        role_play = self.config.get("role_play", [])
        if role_play:
            parts.append(f"=== Role Play Guidelines for {self.name} ===")
            for item in role_play:
                desc = item.get("description", "").strip()
                ex   = item.get("example", "").strip()
                if desc:
                    parts.append(f"- {desc}")
                if ex:
                    parts.append(f'  Example: {ex}')

        # Protective 约束
        protective = self.config.get("protective", [])
        if protective:
            parts.append("=== Protective Guidelines ===")
            for p in protective:
                d = p.get("description", "").strip()
                if d:
                    parts.append(f"- {d}")

        # 任务（使用案件上下文格式化，占位符缺失时回退到原文）
        task = self.config.get("task", "")
        if task:
            formatted_task = self._safe_format(task, self.context)
            parts.append("=== Task ===")
            parts.append(formatted_task.strip())

        # 历史对话
        if self.memory:
            parts.append("=== Conversation So Far ===")
            hist = []
            for m in self.memory:
                speaker = m.get("speaker", "unknown")
                content = m.get("content", "").strip()
                hist.append(f"{speaker}: {content}")
            parts.append("\n".join(hist))

        # 本轮指令
        if task_instruction:
            parts.append("=== Instruction ===")
            parts.append(task_instruction.strip())

        # 要求用角色口吻作答
        parts.append(f"Respond in-character as {self.name}.")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _safe_format(text: str, ctx: Dict[str, Any]) -> str:
        try:
            return text.format(**ctx)
        except Exception:
            return text  # 占位符缺失则原样返回，避免报错中断

    @staticmethod
    def _extract_text(output: Any) -> str:
        if hasattr(output, "content"):
            return output.content
        if isinstance(output, dict):
            return output.get("text") or output.get("content") or str(output)
        return str(output)
