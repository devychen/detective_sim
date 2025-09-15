# collab.py
import os
import random
import time
import yaml
import csv
import datetime   # with time stamp

from agents.holmes_agent import create_agent as create_holmes
from agents.marple_agent import create_agent as create_marple
from agents.poirot_agent import create_agent as create_poirot


class DetectiveDialogue:
    def __init__(self, rule_path="prompts/rule_collab.yaml", case_path="cases/case1.yaml", turns=10):
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

        # ✅ 生成时间戳，用于区分不同 run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # dialogue log
        self.dialogue_log_path = f"data/dialogue_log_{timestamp}.csv"
        with open(self.dialogue_log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["turn", "speaker", "utterance", "believed_murderer"])

        # prompt log 每个 agent 单独一个文件
        self.prompt_log_paths = {}
        for agent_name in self.agents.keys():
            path = f"data/prompt_log_{agent_name}_{timestamp}.txt"
            self.prompt_log_paths[agent_name] = path
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== PROMPT LOG for {agent_name} ===\n\n")

    # -------------------------------
    # helper methods
    # -------------------------------
    def update_memory(self, speaker, content):
        """只存 agent 对话"""
        self.memory.append((speaker, content))

    def get_conversation_history_text(self):
        """返回纯对话形式的 history"""
        history = []
        for speaker, content in self.memory:
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

            current_beliefs = {}  # 存每个 agent 当前的凶手推测

            for agent_name in agent_order:
                agent = self.agents[agent_name]
                agent_prompt = self.agent_prompts[agent_name.lower()]

                # build prompt
                prompt_text = self.build_prompt_for_agent(agent_name, agent_prompt)

                # run
                response = agent.run(prompt_text).strip()

                # 尝试从 response 提取 believed_murderer
                # 假设 agent 的推测格式是 "I believe the murderer is XXX"
                if "murderer is" in response.lower():
                    believed = response.split("murderer is")[-1].strip().split('.')[0]
                else:
                    believed = ""

                current_beliefs[agent_name] = believed

                # update memory
                self.update_memory(agent_name, response)

                # save logs
                self.save_prompt_log(agent_name, prompt_text, t)
                self.save_dialogue_log(t, agent_name, response, believed)

                time.sleep(2)

            # 检查终止条件：三人推测一致
            beliefs_set = set(current_beliefs.values()) - {""}  # 去掉空字符串
            if len(beliefs_set) == 1:
                print(f"All agents agreed on the murderer: {beliefs_set.pop()} at turn {t}")
                break

if __name__ == "__main__":
    sim = DetectiveDialogue()
    sim.simulate()
