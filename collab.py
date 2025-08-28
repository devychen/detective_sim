# collab.py
import os
import csv
import time
import yaml
from agents.holmes_agent import create_agent as create_holmes
from agents.poirot_agent import create_agent as create_poirot
from agents.marple_agent import create_agent as create_marple

class DetectiveDialogue:
    def __init__(self, 
                 rule_path="rules/rule_collab.yaml", 
                 case_path="cases/case1.yaml", 
                 prompt_dir="prompts",
                 turns=10,
                 log_dir="data"):
        self.rules = self._load_yaml(rule_path)
        self.case_info = self._load_yaml(case_path)["case"]
        self.prompt_dir = prompt_dir
        self.turns = turns
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # init agents
        self.agents = {
            "Holmes": create_holmes(),
            "Poirot": create_poirot(),
            "Marple": create_marple(),
        }

        # 用于日志
        self.dialogue_rows = []
        self.prompt_log_file = os.path.join(log_dir, "prompt_log.txt")
        self.dialogue_log_file = os.path.join(log_dir, "dialogue_log.csv")

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_agent_prompt(self, agent_name):
        path = os.path.join(self.prompt_dir, f"{agent_name.lower()}.yaml")
        return self._load_yaml(path)

    def build_prompt(self, agent, context_instruction=""):
        """拼接完整 prompt，包含 system（intro+rules+role+protective+task）+ conversation history"""
        pconf = self._load_agent_prompt(agent.name)

        system_parts = []
        # global intro & rules
        system_parts.append(self.rules["common_intro"].format(agent_name=agent.name))
        system_parts.append(self.rules["common_rules"])
        # agent role_play
        if "role_play" in pconf:
            role_desc = "\n".join([r["description"] for r in pconf["role_play"]])
            system_parts.append(f"Role setting:\n{role_desc}")
        # agent protective
        if "protective" in pconf:
            prot = "\n".join([p["description"] for p in pconf["protective"]])
            system_parts.append(f"Protective constraints:\n{prot}")
        # agent task (with partial clues)
        if "task" in pconf:
            task_filled = pconf["task"].format(
                crime_scene=self.case_info.get("crime_scene", ""),
                forensic_evidence=self.case_info.get("forensic_evidence", "")
            )
            system_parts.append(task_filled)

        # conversation so far
        conv_history = agent.get_memory_text()

        # 最终拼接
        full_prompt = (
            "=== System Context ===\n" 
            + "\n\n".join(system_parts)
            + "\n\n"
            + conv_history
        )
        if context_instruction:
            full_prompt += f"\n\nInstruction: {context_instruction}"

        # 保存 prompt log
        with open(self.prompt_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== Prompt for {agent.name} =====\n{full_prompt}\n")

        return full_prompt

    def broadcast(self, speaker, content):
        """把某个 agent 的发言广播到其他人"""
        for name, agent in self.agents.items():
            if name != speaker.name:
                agent.update_memory(speaker.name, content)

    def run_dialogue(self):
        for turn in range(self.turns):
            for agent in self.agents.values():
                task_instruction = "Continue the collaborative investigation."
                full_prompt = self.build_prompt(agent, task_instruction)
                response = agent.run(full_prompt)

                # 保存到 memory
                agent.update_memory(agent.name, response)
                self.broadcast(agent, response)

                # 提取嫌疑人（简单匹配）
                suspect = self._extract_suspect(response)

                # 存 log row
                self.dialogue_rows.append([
                    turn+1, agent.name, response, suspect
                ])

                time.sleep(2)  # 避免超速调用

            # 检查是否达成一致
            suspects = [row[3] for row in self.dialogue_rows if row[0] == turn+1]
            if all(suspects) and len(set(suspects)) == 1 and suspects[0].lower() not in [
                "unknown", "undetermined", "not sure", "uncertain"
            ]:
                break

        self.save_log()

    def _extract_suspect(self, text: str) -> str:
        marker = "I believe the murderer is:"
        if marker in text:
            return text.split(marker)[-1].strip().split()[0]
        return "Uncertain"

    def save_log(self):
        with open(self.dialogue_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
            writer.writerows(self.dialogue_rows)
