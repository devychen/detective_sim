# collab.py
import yaml
import csv
import os
import time
from agents.holmes_agent import HolmesAgent
from agents.poirot_agent import PoirotAgent
from agents.marple_agent import MarpleAgent


class DetectiveDialogue:
    def __init__(self, rule_path="rules/rule_collab.yaml", case_path="cases/case1.yaml", turns=10):
        self.rule_path = rule_path
        self.case_path = case_path
        self.rules = self._load_yaml(rule_path)
        self.case_info = self._load_yaml(case_path)
        self.turns = turns

        # 输出目录
        os.makedirs("data", exist_ok=True)
        self.log_file = "data/dialogue_log.csv"
        self.prompt_log_file = "data/prompt_log.txt"

        # 初始化三个探员
        self.agents = [
            HolmesAgent(),
            PoirotAgent(),
            MarpleAgent()
        ]

        # 给每个 agent 注入 system prompt（但不进入 conversation history）
        for agent in self.agents:
            agent.set_system_prompt([
                agent.get_role_play_text(),
                agent.get_protective_text(),
                self.rules["common_intro"].format(agent_name=agent.name),
                self.rules["common_rules"],
            ])
            agent.set_initial_clues(self.case_info.get(agent.name.lower(), {}))  # 每个 agent 自己的线索

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def format_input_for_agent(self, agent, context_instruction=""):
        """构造 agent 当前 step 的完整输入 prompt"""
        system_prompt = agent.system_prompt  # 独立存放，不混入 history

        history = agent.get_conversation_history_text()  # 仅 agent 间对话

        # 拼接任务：初始线索 + 动态指令
        task = agent.get_task_prompt()
        if context_instruction:
            task += f"\n{context_instruction}"

        full_prompt = (
            f"{system_prompt}\n"
            f"=== Conversation So Far ===\n{history}\n"
            f"=== Current Task ===\n{task}"
        )
        return full_prompt

    def broadcast(self, speaker, content):
        """把某个 agent 的发言广播到其他人"""
        for agent in self.agents:
            if agent.name != speaker.name:
                agent.update_memory(speaker.name, content)

    def extract_suspect(self, response: str):
        """从 agent 回复中提取结论里的嫌疑人"""
        marker = "I believe the murderer is:"
        if marker in response:
            return response.split(marker)[-1].strip().split()[0]
        return None

    def run_dialogue(self):
        dialogue_rows = []
        prompt_logs = []
        suspects = []

        for turn in range(self.turns):
            for agent in self.agents:
                task_instruction = self.format_input_for_agent(agent)

                # 保存 prompt 到日志
                prompt_logs.append(
                    f"\n=== Turn {turn+1}, Agent: {agent.name} ===\n{task_instruction}\n"
                )

                response = agent.run(task_instruction)
                self.broadcast(agent, response)

                suspected = self.extract_suspect(response)
                if suspected:
                    suspects.append(suspected)

                dialogue_rows.append([turn+1, agent.name, response, suspected])

                time.sleep(2)  # 防止超过 RPM 限制

            # 检查提前停止条件（全体一致）
            if suspects and len(suspects) >= len(self.agents):
                last_round = suspects[-len(self.agents):]
                if all(last_round) and len(set(last_round)) == 1 and last_round[0].lower() not in [
                    "unknown", "undetermined", "not sure", "uncertain"
                ]:
                    break

        # 存日志
        self.save_log(dialogue_rows)
        self.save_prompt_log(prompt_logs)

    def save_log(self, dialogue_rows):
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
            writer.writerows(dialogue_rows)

    def save_prompt_log(self, prompt_logs):
        with open(self.prompt_log_file, "w", encoding="utf-8") as f:
            for row in prompt_logs:
                f.write(row + "\n")
