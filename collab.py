# collab.py
import yaml
import csv
import os
import time
from agents.holmes_agent import HolmesAgent
from agents.poirot_agent import PoirotAgent
from agents.marple_agent import MarpleAgent


class DetectiveDialogue:
    def __init__(self, 
                 rule_path="rules/rule_collab.yaml", 
                 case_path="cases/case1.yaml", 
                 log_file="data/dialogue_log.csv", 
                 turns=10):
        self.rule_path = rule_path
        self.case_path = case_path
        self.rules = self._load_yaml(rule_path)
        self.case_info = self._load_yaml(case_path)
        self.turns = turns
        self.log_file = log_file

        # 初始化三个探员
        self.agents = [
            HolmesAgent(),
            PoirotAgent(),
            MarpleAgent()
        ]

        # 给每个 agent 注入开场提示和全局规则
        for agent in self.agents:
            agent.update_memory("system", self.rules["common_intro"].format(agent_name=agent.name))
            agent.update_memory("system", self.rules["common_rules"])
            agent.update_memory("system", f"Case info: {self.case_info}")

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def format_input_for_agent(self, agent, context_instruction=""):
        """构造 agent 当前 step 的输入"""
        base_task = (
            "Continue the collaborative investigation.\n"
            "Remember to strictly follow the common rules and your protective constraints."
        )
        if context_instruction:
            base_task += f"\n{context_instruction}"
        return base_task

    def broadcast(self, speaker, content):
        """把某个 agent 的发言广播到其他人"""
        for agent in self.agents:
            if agent.name != speaker.name:
                agent.update_memory(speaker.name, content)

    def extract_suspect(self, text: str):
        """
        从 agent 的发言里提取结论。
        格式要求: 'I believe the murderer is: XXX'
        """
        marker = "I believe the murderer is:"
        if marker in text:
            return text.split(marker, 1)[1].strip().split()[0]  # 取名字部分
        return None

    def save_log(self, dialogue_rows):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
            writer.writerows(dialogue_rows)

    def run_dialogue(self):
        dialogue_rows = []
        suspects = {agent.name: None for agent in self.agents}

        for turn in range(self.turns):
            for agent in self.agents:
                task_instruction = self.format_input_for_agent(agent)
                response = agent.run(task_instruction)
                time.sleep(2)  # 避免触发速率限制

                # 提取 suspect
                murderer_guess = self.extract_suspect(response)
                if murderer_guess:
                    suspects[agent.name] = murderer_guess

                dialogue_rows.append([turn + 1, agent.name, response, murderer_guess or "N/A"])

                # 广播给其他人
                self.broadcast(agent, response)

            # 检查是否三人达成一致
            current_guesses = [s for s in suspects.values() if s]
            if len(current_guesses) == len(self.agents):
                if len(set(g.lower() for g in current_guesses)) == 1:
                    agreed = current_guesses[0].lower()
                    if agreed not in ["unknown", "undetermined", "not sure", "uncertain"]:
                        # 三人达成一致 → 停止
                        self.save_log(dialogue_rows)
                        return

        # 如果没提前停止 → 存储全局对话
        self.save_log(dialogue_rows)
