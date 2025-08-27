# collab.py
import time
from datetime import datetime
import yaml
import os
import random
import re
import csv

from agent._archive.holmes_agent import create_holmes_agent
from agent._archive.poirot_agent import create_poirot_agent
from agent._archive.marple_agent import create_marple_agent

class DetectiveDialogue:
    def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.csv", case_file="cases/case1.yaml"):
        self.max_turns = max_turns
        self.delay = delay
        self.log_file = log_file
        self.memory = {name: [] for name in ["Holmes", "Poirot", "Marple"]}

        # Load case file
        if not os.path.exists(case_file):
            raise FileNotFoundError(f"Case file not found: {case_file}")
        with open(case_file, 'r', encoding='utf-8') as f:
            self.case_data = yaml.safe_load(f)["case"]

        # Load rules and protective prompts
        with open("rules/rule_collab.yaml", 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)
        with open("prompts/protective.yaml", 'r', encoding='utf-8') as f:
            self.protectives = yaml.safe_load(f)

        # Agents
        self.agents = {
            "Holmes": create_holmes_agent(),
            "Poirot": create_poirot_agent(),
            "Marple": create_marple_agent(),
        }

    def format_input_for_agent(self, agent_name: str) -> str:
        history_parts = [
            f"【{name}】said:\n" + "\n".join(messages[-3:])
            for name, messages in self.memory.items() if messages
        ]
        chat_history = "\n\n".join(history_parts)

        context = {
            "agent_name": agent_name,
            "setting": self.case_data["setting"],
            "victim": self.case_data["victim"],
            "crime_scene": self.case_data.get("crime_scene", {}),
            "forensic_evidence": self.case_data.get("forensic_evidence", {}),
            "timeline": self.case_data.get("timeline", {}),
            "suspects": self.case_data.get("suspects", {}),
            "protective_holmes": self.protectives.get("protective_holmes", []),
            "protective_poirot": self.protectives.get("protective_poirot", []),
            "protective_marple": self.protectives.get("protective_marple", []),
        }

        rule_prompt = self.rules[f"system_prompt_{agent_name.lower()}"]
        formatted_prompt = rule_prompt.format(**context)

        return f"""
        {formatted_prompt}

        【Shared Dialogue History】:
        {chat_history if chat_history else "(None yet)"}
        """.strip()

    def extract_suspect_from(self, response: str) -> str | None:
        patterns = [
            r"I believe the murderer is[:：]\s*(\w+)",
            r"The killer (?:must be|is)\s*[:：]?\s*(\w+)",
            r"(?:murderer|killer)\s*is\s*(\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def save_log(self, dialogue_rows):
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Turn No.", "Agent Name", "Spoken Content", "Believed murderer"])
            writer.writerows(dialogue_rows)

    def run_dialogue(self):
        dialogue_rows = []
        for turn in range(self.max_turns):
            agent_order = ["Holmes", "Poirot", "Marple"]
            random.shuffle(agent_order)

            responses = {}
            for agent_name in agent_order:
                agent = self.agents[agent_name]
                prompt = self.format_input_for_agent(agent_name)
                response = agent.run(prompt)
                self.memory[agent_name].append(response)
                responses[agent_name] = response

                believed = self.extract_suspect_from(response) or ""
                dialogue_rows.append([turn + 1, agent_name, response.strip(), believed])

                for other_agent_name in self.agents:
                    if other_agent_name != agent_name:
                        self.agents[other_agent_name].update_memory(agent_name, response)

                time.sleep(self.delay)

            suspects = [self.extract_suspect_from(responses[a]) for a in ["Holmes", "Poirot", "Marple"]]
            if all(suspects) and len(set(suspects)) == 1 and suspects[0].lower() not in ["unknown", "undetermined", "not sure", "uncertain"]:
                break

        self.save_log(dialogue_rows)



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
