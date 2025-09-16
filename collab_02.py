# collab.py
import os
import re
import random
import time
import yaml
import csv
import datetime   # with time stamp

from agents.holmes_agent import create_agent as create_holmes
from agents.marple_agent import create_agent as create_marple
from agents.poirot_agent import create_agent as create_poirot


def extract_believed_murderer(response: str) -> str:
    """
    Extract the suspected murderer's name from an agent's response.
    Returns cleaned name (without leading punctuation), or "" if not found.
    """
    text = response.strip()

    patterns = [
        r"(?:murderer is|killer is|I think it(?:'s| is)|I believe(?: it)?(?:'s| is))\s+([A-Z][a-zA-Z]+)",
        r"(?:suspect is|culprit is)\s+([A-Z][a-zA-Z]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1)
            return name.strip("：:,.，。!? ")

    return ""


class DetectiveDialogue:
    def __init__(self, rule_path="prompts/rule_collab.yaml", case_path="cases/case2.yaml", turns=10):
        self.turns = turns
        self.memory = []  # [(speaker, content)]

        # load rules
        with open(rule_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

        # load case
        with open(case_path, "r", encoding="utf-8") as f:
            self.case = yaml.safe_load(f)["case"]

        # load agents
        self.agents = {
            "Holmes": create_holmes(),
            "Marple": create_marple(),
            "Poirot": create_poirot(),
        }

        # load agent prompts (role_play + protective + task)
        self.agent_prompts = {}
        for name in ["holmes", "marple", "poirot"]:
            path = f"prompts/{name}.yaml"
            with open(path, "r", encoding="utf-8") as f:
                self.agent_prompts[name] = yaml.safe_load(f)

        # make sure data dir exists
        os.makedirs("data", exist_ok=True)

        # 生成本次 run 的独立文件夹（带 timestamp）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("data", f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # dialogue log
        self.dialogue_log_path = os.path.join(self.run_dir, "dialogue_log.csv")
        with open(self.dialogue_log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["turn", "speaker", "utterance", "believed_murderer"])

        # prompt log 每个 agent 单独一个文件
        self.prompt_log_paths = {}
        for agent_name in self.agents.keys():
            path = os.path.join(self.run_dir, f"prompt_log_{agent_name}.txt")
            self.prompt_log_paths[agent_name] = path
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== PROMPT LOG for {agent_name} ===\n\n")

    # -------------------------------
    # helper methods
    # -------------------------------
    def update_memory(self, speaker, content):
        """只存 agent 对话"""
        self.memory.append((speaker, content))

    def get_conversation_history_text(self, max_turns=5):
        """返回最近 max_turns 轮的对话"""
        history = []
        for speaker, content in self.memory[-max_turns:]:
            history.append(f"{speaker}: {content}")
        return "\n".join(history)

    def get_partial_clues(self, agent_name):
        """返回某个 agent 对应的 partial case info"""
        if agent_name == "Holmes":
            return {
                "crime_scene": self.case.get("crime_scene", {}),
                "forensic_evidence": self.case.get("forensic_evidence", {}),
            }
        elif agent_name == "Marple":
            return {
                "suspects": self.case.get("suspects", []),
            }
        elif agent_name == "Poirot":
            return {
                "timeline": self.case.get("timeline", []),
            }
        else:
            return {}

    def build_system_prompt(self, agent_name, agent_prompt):
        """拼接 system prompt（不进 memory）"""
        role_play = "\n".join(
            [f"- {r['description']}" for r in agent_prompt.get("role_play", [])]
        )
        protective = "\n".join(
            [f"- {p['description']}" for p in agent_prompt.get("protective", [])]
        )

        # partial clues
        partial_clues = self.get_partial_clues(agent_name)
        task_template = agent_prompt.get("task", "")
        task_text = task_template.format(**partial_clues)

        system_prompt = f"""You are {agent_name}.

=== Background ===
{self.rules.get('common_intro', '').format(agent_name=agent_name)}

=== Collaboration Rules ===
{self.rules.get('common_rules', '')}

=== Role Play Guidelines ===
{role_play}

=== Protective Guidelines ===
{protective}

=== Task ===
{task_text}
        """
        return system_prompt

    def build_prompt_for_agent(self, agent_name, agent_prompt):
        """拼接完整输入：system + conversation so far"""
        system_prompt = self.build_system_prompt(agent_name, agent_prompt)
        history_text = self.get_conversation_history_text()

        full_prompt = f"""{system_prompt}

=== Conversation So Far ===
{history_text if history_text else "(no conversation yet)"}
        """
        return full_prompt

    def save_prompt_log(self, agent_name, prompt_text, turn):
        """保存每个 agent 自己的 prompt"""
        path = self.prompt_log_paths[agent_name]
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- Turn {turn}, Agent: {agent_name} ---\n")
            f.write(prompt_text)
            f.write("\n\n")

    def save_dialogue_log(self, turn, speaker, utterance, believed_murderer=""):
        """增加 believed_murderer 列"""
        with open(self.dialogue_log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([turn, speaker, utterance, believed_murderer])

    # -------------------------------
    # main simulation
    # -------------------------------
    def simulate(self):
        for t in range(1, self.turns + 1):
            agent_order = list(self.agents.keys())
            random.shuffle(agent_order)

            current_beliefs = {}

            for agent_name in agent_order:
                agent = self.agents[agent_name]
                agent_prompt = self.agent_prompts[agent_name.lower()]

                # build prompt
                prompt_text = self.build_prompt_for_agent(agent_name, agent_prompt)

                # run
                response = agent.run(prompt_text).strip()

                # 提取 believed_murderer
                believed = extract_believed_murderer(response)
                current_beliefs[agent_name] = believed

                # update memory
                self.update_memory(agent_name, response)

                # save logs
                self.save_prompt_log(agent_name, prompt_text, t)
                self.save_dialogue_log(t, agent_name, response, believed)

                time.sleep(2)

            # 检查终止条件：三人都有推测，且一致
            beliefs = list(current_beliefs.values())
            if all(beliefs) and len(set(beliefs)) == 1:
                print(f"All agents agreed on the murderer: {beliefs[0]} at turn {t}")
                break


if __name__ == "__main__":
    sim = DetectiveDialogue()
    sim.simulate()
