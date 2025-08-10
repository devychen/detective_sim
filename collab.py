# collab.py
import time
import yaml
import os
import random
import re
import csv

from agents.holmes_agent import create_holmes_agent
from agents.poirot_agent import create_poirot_agent
from agents.marple_agent import create_marple_agent

class DetectiveDialogue:
    def __init__(self, max_turns=10, delay=1, run_id="001",
                 data_dir="data", case_file="cases/case1.yaml"):
        self.max_turns = max_turns
        self.delay = delay
        self.run_id = run_id
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # 自动生成文件路径，便于统一管理
        self.csv_file = os.path.join(self.data_dir, f"collab_run_{self.run_id}.csv")
        self.prompt_yaml_file = os.path.join(self.data_dir, f"collab_run_{self.run_id}_prompts.yaml")

        self.memory = {name: [] for name in ["Holmes", "Poirot", "Marple"]}

        # 加载案件及配置
        if not os.path.exists(case_file):
            raise FileNotFoundError(f"Case file not found: {case_file}")
        with open(case_file, 'r', encoding='utf-8') as f:
            self.case_data = yaml.safe_load(f)["case"]

        with open("rules/rule_collab.yaml", 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

        with open("prompts/protective.yaml", 'r', encoding='utf-8') as f:
            self.protectives = yaml.safe_load(f)

        # 初始化三个侦探代理，传入任务线索和保护规则
        self.agents = {
            "Holmes": create_holmes_agent(
                clues={**self.case_data.get("crime_scene", {}), **self.case_data.get("forensic_evidence", {})},
                protective=self.protectives.get("protective_holmes", [])
            ),
            "Poirot": create_poirot_agent(
                clues=self.case_data.get("timeline", {}),
                protective=self.protectives.get("protective_poirot", [])
            ),
            "Marple": create_marple_agent(
                clues=self.case_data.get("suspects", {}),
                protective=self.protectives.get("protective_marple", [])
            ),
        }

        # 用于存储每个agent最终生成的prompt
        self.final_prompts = {}

    def format_input_for_agent(self, agent_name: str) -> str:
        history = []
        for name, messages in self.memory.items():
            if messages:
                history.append(f"【{name}】said:\n" + "\n".join(messages[-3:]))
        chat_history = "\n\n".join(history) if history else "(None yet)"

        context = {
            "agent_name": agent_name,
            "setting": self.case_data.get("setting", ""),
            "victim": self.case_data.get("victim", ""),
            "crime_scene": self.case_data.get("crime_scene", {}),
            "forensic_evidence": self.case_data.get("forensic_evidence", {}),
            "timeline": self.case_data.get("timeline", {}),
            "suspects": self.case_data.get("suspects", {}),
            "protective_holmes": self.protectives.get("protective_holmes", []),
            "protective_poirot": self.protectives.get("protective_poirot", []),
            "protective_marple": self.protectives.get("protective_marple", []),
        }

        system_prompt = self.rules.get("common_intro", "") + "\n" + self.rules.get("common_rules", "")
        if agent_name == "Holmes":
            system_prompt += "\n" + self.rules.get("holmes_task", "").format(**context)
        elif agent_name == "Poirot":
            system_prompt += "\n" + self.rules.get("poirot_task", "").format(**context)
        elif agent_name == "Marple":
            system_prompt += "\n" + self.rules.get("marple_task", "").format(**context)

        prompt = f"""{system_prompt}

【Shared Dialogue History】:
{chat_history}
"""
        return prompt.strip()

    def extract_suspect_from(self, text: str) -> str | None:
        pattern = r"I believe the murderer is[:：]\s*([\w\s\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def save_log(self, rows):
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Turn", "Agent", "Utterance", "Suspect"])
            writer.writerows(rows)

    def save_prompts_yaml(self):
        # 保存所有agent最终使用的prompt到yaml文件
        with open(self.prompt_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.final_prompts, f, allow_unicode=True)

    def run_dialogue(self):
        dialogue_rows = []
        for turn in range(1, self.max_turns + 1):
            agent_order = random.sample(list(self.agents.keys()), 3)

            for agent_name in agent_order:
                prompt = self.format_input_for_agent(agent_name)
                response = self.agents[agent_name].run(prompt).strip()
                self.memory[agent_name].append(response)

                suspect = self.extract_suspect_from(response) or ""
                dialogue_rows.append([turn, agent_name, response, suspect])

                for other_name, other_agent in self.agents.items():
                    if other_name != agent_name:
                        other_agent.update_memory(agent_name, response)

                # 记录最后一次的prompt，覆盖更新
                self.final_prompts[agent_name] = prompt

                time.sleep(self.delay)

            suspects = [self.extract_suspect_from(self.memory[agent][-1]) for agent in self.agents]
            suspects_lower = [s.lower() for s in suspects if s]
            if (len(set(suspects_lower)) == 1 and suspects_lower and
                suspects_lower[0] not in ["unknown", "undetermined", "not sure", "uncertain"]):
                break

        self.save_log(dialogue_rows)
        self.save_prompts_yaml()


# # collab.py

# import time
# from datetime import datetime
# import yaml
# import os
# import random
# import re
# import csv

# from agents.holmes_agent import create_holmes_agent
# from agents.poirot_agent import create_poirot_agent
# from agents.marple_agent import create_marple_agent

# class DetectiveDialogue:
#     def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.csv", case_file="cases/case1.yaml"):
#         self.max_turns = max_turns
#         self.delay = delay
#         self.log_file = log_file
#         self.memory = {name: [] for name in ["Holmes", "Poirot", "Marple"]}

#         # Load case file
#         if not os.path.exists(case_file):
#             raise FileNotFoundError(f"Case file not found: {case_file}")
#         with open(case_file, 'r', encoding='utf-8') as f:
#             self.case_data = yaml.safe_load(f)["case"]

#         # Load rules and protective prompts
#         with open("rules/rule_collab.yaml", 'r', encoding='utf-8') as f:
#             self.rules = yaml.safe_load(f)
#         with open("prompts/protective.yaml", 'r', encoding='utf-8') as f:
#             self.protectives = yaml.safe_load(f)

#         # Agents
#         self.agents = {
#             "Holmes": create_holmes_agent(),
#             "Poirot": create_poirot_agent(),
#             "Marple": create_marple_agent(),
#         }

#     def format_input_for_agent(self, agent_name: str) -> str:
#         history_parts = [
#             f"【{name}】said:\n" + "\n".join(messages[-3:])
#             for name, messages in self.memory.items() if messages
#         ]
#         chat_history = "\n\n".join(history_parts)

#         context = {
#             "agent_name": agent_name,
#             "setting": self.case_data["setting"],
#             "victim": self.case_data["victim"],
#             "crime_scene": self.case_data.get("crime_scene", {}),
#             "forensic_evidence": self.case_data.get("forensic_evidence", {}),
#             "timeline": self.case_data.get("timeline", {}),
#             "suspects": self.case_data.get("suspects", {}),
#             "protective_holmes": self.protectives.get("protective_holmes", []),
#             "protective_poirot": self.protectives.get("protective_poirot", []),
#             "protective_marple": self.protectives.get("protective_marple", []),
#         }

#         rule_prompt = self.rules[f"system_prompt_{agent_name.lower()}"]
#         formatted_prompt = rule_prompt.format(**context)

#         return f"""
#         {formatted_prompt}

#         【Shared Dialogue History】:
#         {chat_history if chat_history else "(None yet)"}
#         """.strip()

#     def extract_suspect_from(self, response: str) -> str | None:
#         patterns = [
#             r"I believe the murderer is[:：]\s*(\w+)",
#             r"The killer (?:must be|is)\s*[:：]?\s*(\w+)",
#             r"(?:murderer|killer)\s*is\s*(\w+)",
#         ]
#         for pattern in patterns:
#             match = re.search(pattern, response, re.IGNORECASE)
#             if match:
#                 return match.group(1).strip()
#         return None

#     def save_log(self, dialogue_rows):
#         with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
#             writer.writerows(dialogue_rows)

#     def run_dialogue(self):
#         dialogue_rows = []
#         for turn in range(self.max_turns):
#             agent_order = ["Holmes", "Poirot", "Marple"]
#             random.shuffle(agent_order)

#             responses = {}
#             for agent_name in agent_order:
#                 agent = self.agents[agent_name]
#                 prompt = self.format_input_for_agent(agent_name)
#                 response = agent.run(prompt)
#                 self.memory[agent_name].append(response)
#                 responses[agent_name] = response

#                 believed = self.extract_suspect_from(response) or ""
#                 dialogue_rows.append([turn + 1, agent_name, response.strip(), believed])

#                 for other_agent_name in self.agents:
#                     if other_agent_name != agent_name:
#                         self.agents[other_agent_name].update_memory(agent_name, response)

#                 time.sleep(self.delay)

#             suspects = [self.extract_suspect_from(responses[a]) for a in ["Holmes", "Poirot", "Marple"]]
#             if all(suspects) and len(set(suspects)) == 1 and suspects[0].lower() not in ["unknown", "undetermined", "not sure", "uncertain"]:
#                 break

#         self.save_log(dialogue_rows)
