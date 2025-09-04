# collab.py
# 支持流式响应（避免 504 超时）。
# CSV 增加 believed_murderer 列。
# 停止条件：达到 10 轮 或 三个侦探意见一致。
# max_tokens 默认不传，保持旧行为，只有在调用时显式传才生效。

import os
import random
import time
import yaml
import csv
import datetime
import re

from agents.holmes_agent import create_agent as create_holmes
from agents.marple_agent import create_agent as create_marple
from agents.poirot_agent import create_agent as create_poirot


class DetectiveDialogue:
    
    def __init__(self, rule_path="prompts/rule_collab.yaml", case_path="cases/case3.yaml", turns=10, max_tokens=None, temperature=None):
        self.turns = turns
        self.max_tokens = max_tokens  # 默认 None
        self.temperature = temperature  # 默认 None，不传给 API
        self.memory = []  # [(speaker, content, believed_murderer)]


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

        # timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dialogue_log_path = f"data/dialogue_log_{timestamp}.csv"
        self.prompt_log_path = f"data/prompt_log_{timestamp}.txt"

        # reset logs
        with open(self.dialogue_log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["turn", "speaker", "utterance", "believed_murderer"])
        with open(self.prompt_log_path, "w", encoding="utf-8") as f:
            f.write("=== PROMPT LOG ===\n\n")

        # 🔍 打印当前配置
        print(f"[DetectiveDialogue] max_tokens={self.max_tokens}, temperature={self.temperature}")

    # -------------------------------
    # helper methods
    # -------------------------------
    def update_memory(self, speaker, content, believed_murderer):
        self.memory.append((speaker, content, believed_murderer))

    def get_conversation_history_text(self):
        history = []
        for speaker, content, _ in self.memory:
            history.append(f"{speaker}: {content}")
        return "\n".join(history)

    def get_partial_clues(self, agent_name):
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
        role_play = "\n".join(
            [f"- {r['description']}" for r in agent_prompt.get("role_play", [])]
        )
        protective = "\n".join(
            [f"- {p['description']}" for p in agent_prompt.get("protective", [])]
        )

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
        system_prompt = self.build_system_prompt(agent_name, agent_prompt)
        history_text = self.get_conversation_history_text()

        full_prompt = f"""{system_prompt}

=== Conversation So Far ===
{history_text if history_text else "(no conversation yet)"}
"""
        return full_prompt

    def save_prompt_log(self, agent_name, prompt_text, turn):
        with open(self.prompt_log_path, "a", encoding="utf-8") as f:
            f.write(f"--- Turn {turn}, Agent: {agent_name} ---\n")
            f.write(prompt_text)
            f.write("\n\n")

    def save_dialogue_log(self, turn, speaker, utterance, believed_murderer=None):
        with open(self.dialogue_log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([turn, speaker, utterance, believed_murderer])

    def extract_believed_murderer(self, text):
        """简单正则提取 'murderer is X'"""
        match = re.search(r"(?:murderer|culprit).{0,20}?(\b[A-Z][a-zA-Z]+\b)", text)
        if match:
            return match.group(1)
        return None

    def check_agreement(self):
        """检查最后三句话是否意见一致"""
        beliefs = [bm for _, _, bm in self.memory if bm]
        if len(beliefs) >= 3 and len(set(beliefs[-3:])) == 1:
            return True
        return False

    # -------------------------------
    # main simulation
    # -------------------------------
    def simulate(self):
        for t in range(1, self.turns + 1):
            agent_order = list(self.agents.keys())
            random.shuffle(agent_order)

            for agent_name in agent_order:
                agent = self.agents[agent_name]
                agent_prompt = self.agent_prompts[agent_name.lower()]

                # build prompt
                prompt_text = self.build_prompt_for_agent(agent_name, agent_prompt)

                # run (支持 max_tokens，但默认不传)
                kwargs = {"stream": True}
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature

                response = agent.run(prompt_text, **kwargs).strip()


                # extract believed murderer
                believed_murderer = self.extract_believed_murderer(response)

                # update memory
                self.update_memory(agent_name, response, believed_murderer)

                # save logs
                self.save_prompt_log(agent_name, prompt_text, t)
                self.save_dialogue_log(t, agent_name, response, believed_murderer)

            # check stop condition
            if self.check_agreement():
                break

            time.sleep(2)  # 每轮休眠，避免速率限制



if __name__ == "__main__":
    # 默认行为（不传参数）
    # sim = DetectiveDialogue()

    # 控制回复长度
    # sim = DetectiveDialogue(max_tokens=400)

    # 控制创造性
    # sim = DetectiveDialogue(temperature=0.7)

    # 同时控制
    # sim = DetectiveDialogue(max_tokens=400, temperature=0.7)

    sim = DetectiveDialogue()
    sim.simulate()
