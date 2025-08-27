# collab.py
import os
import time
import re
import csv
import yaml
from typing import List, Dict, Optional
from datetime import datetime

from agents.holmes_agent import HolmesAgent
from agents.poirot_agent import PoirotAgent
from agents.marple_agent import MarpleAgent


class DetectiveDialogue:
    """
    协作管理器：加载规则/案件；初始化三位探员；轮次管理；共识停止；日志与 Prompt 落盘。
    """
    def __init__(
        self,
        rule_path: str = "rules/rule_collab.yaml",
        case_path: str = "cases/case2.yaml",
        turns: int = 10,
        log_file: str = "data/dialogue_log.csv",
        prompt_file: str = "data/prompts_log.txt",
        append_timestamp: bool = False,
    ):
        self.turns = turns
        self.rules = self._load_yaml(rule_path)
        self.case_ctx = self._load_case(case_path)

        # 输出文件：可选时间戳，避免覆盖
        if append_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_root, log_ext = os.path.splitext(log_file)
            prm_root, prm_ext = os.path.splitext(prompt_file)
            log_file  = f"{log_root}_{ts}{log_ext or '.csv'}"
            prompt_file = f"{prm_root}_{ts}{prm_ext or '.txt'}"

        self.log_file = log_file
        self.prompt_file = prompt_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.prompt_file), exist_ok=True)

        # 初始化三个探员（你已有的 Poirot/Marple 类保持不变）
        self.agents = [
            HolmesAgent(),
            PoirotAgent(),
            MarpleAgent(),
        ]

        # 注入全局 intro/rules + 案件上下文
        for agent in self.agents:
            agent.set_context(self.case_ctx)  # 供 task.format(**ctx) 使用
            agent.update_memory("system", self.rules.get("common_intro", "").format(agent_name=agent.name))
            agent.update_memory("system", self.rules.get("common_rules", ""))
            # 把案件信息以 YAML 格式写入系统记忆，便于 LLM 消化
            # agent.update_memory("system", "Case Information:\n" + yaml.safe_dump(self.case_ctx, allow_unicode=True))

    # ---------- 文件加载 ----------
    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_case(path: str) -> Dict:
        data = DetectiveDialogue._load_yaml(path)
        # 兼容两种结构：{case: {...}} 或直接 {...}
        return data.get("case", data)

    # ---------- Prompt 组装（回合级别的额外指令） ----------
    @staticmethod
    def format_input_for_agent(agent_name: str, extra: str = "") -> str:
        base = (
            "Continue the collaborative investigation.\n"
            "Remember to strictly follow the common rules and your protective constraints."
        )
        if extra:
            base += f"\n{extra}"
        return base

    # ---------- 广播 ----------
    def broadcast(self, speaker_name: str, content: str) -> None:
        for agent in self.agents:
            if agent.name != speaker_name:
                agent.update_memory(speaker_name, content)

    # ---------- 提取凶手名 ----------
    @staticmethod
    def extract_suspect(text: str) -> Optional[str]:
        """
        解析结论格式（尽量鲁棒）：
        - I believe the murderer is: XXX
        - The killer is XXX
        - Murderer is XXX
        """
        patterns = [
            r"I believe the murderer is[:：]\s*([A-Za-z][A-Za-z\-\.' ]+)",
            r"The killer (?:must be|is)\s*[:：]?\s*([A-Za-z][A-Za-z\-\.' ]+)",
            r"(?:murderer|killer)\s*is\s*([A-Za-z][A-Za-z\-\.' ]+)",
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                guess = m.group(1).strip()
                # 截断到行尾/句号前，避免抓到多余话
                guess = re.split(r"[\n\r\.,;!?:]", guess)[0].strip()
                return guess if guess else None
        return None

    # ---------- 日志 ----------
    def save_log(self, rows: List[List[str]]) -> None:
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
            writer.writerows(rows)

    def append_prompt_dump(self, turn: int, agent_name: str, prompt: str, response: str) -> None:
        with open(self.prompt_file, "a", encoding="utf-8") as f:
            f.write(f"--- Round {turn} | {agent_name} ---\n")
            f.write("[INPUT PROMPT]\n")
            f.write(prompt.strip() + "\n\n")
            f.write("[RESPONSE]\n")
            f.write(response.strip() + "\n\n")

    # ---------- 主流程 ----------
    def run_dialogue(self) -> None:
        dialogue_rows: List[List[str]] = []

        for turn in range(1, self.turns + 1):
            turn_suspects: List[str] = []

            # 固定顺序（如需随机，可自行打乱 self.agents）
            for agent in self.agents:
                instruction = self.format_input_for_agent(agent.name)
                response, prompt = agent.run(instruction)

                # 记录
                guess = self.extract_suspect(response) or ""
                dialogue_rows.append([turn, agent.name, response, guess or ""])
                self.append_prompt_dump(turn, agent.name, prompt, response)

                # 广播给其他探员
                self.broadcast(agent.name, response)

                # 每个 agent 说完等 2 秒，避免触发 RPM 限制
                time.sleep(3)

                if guess:
                    turn_suspects.append(guess)

            # 停止条件：三人一致且不为 unknown 等
            if len(turn_suspects) == len(self.agents):
                lowered = [g.lower() for g in turn_suspects]
                if len(set(lowered)) == 1 and lowered[0] not in {"unknown", "undetermined", "not sure", "uncertain"}:
                    self.save_log(dialogue_rows)
                    return

        # 若未提前一致，保存满回合日志
        self.save_log(dialogue_rows)
