
import time
from datetime import datetime
import yaml
import os
import random
import re

from agents._archive.holmes_agent import create_holmes_agent
from agents._archive.poirot_agent import create_poirot_agent
from agents._archive.marple_agent import create_marple_agent

class DetectiveDialogue:
    def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt", case_file="cases/case1.yaml"):
        self.max_turns = max_turns
        self.delay = delay
        self.log_file = log_file
        self.log_lines = []

        # Load case file
        if not os.path.exists(case_file):
            raise FileNotFoundError(f"Case file not found: {case_file}")

        with open(case_file, 'r', encoding='utf-8') as f:
            self.case_data = yaml.safe_load(f)["case"]

        # Load rule prompts
        with open("rules/rule.yaml", 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

        # Load protective prompts
        with open("protective_prompts/protective.yaml", 'r', encoding='utf-8') as f:
            self.protectives = yaml.safe_load(f)

        # Initialize memory
        self.memory = {
            "Holmes": [],
            "Poirot": [],
            "Marple": [],
        }

        # Initialize agents
        self.holmes = create_holmes_agent()
        self.poirot = create_poirot_agent()
        self.marple = create_marple_agent()

        self.agents = {
            "Holmes": self.holmes,
            "Poirot": self.poirot,
            "Marple": self.marple,
        }

    def format_input_for_agent(self, agent_name: str) -> str:
        history_parts = []
        for name, messages in self.memory.items():
            if messages:
                history_parts.append(f"【{name}】said:\n" + "\n".join(messages[-3:]))

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

        # final_prompt = f"""
        # {formatted_prompt}

        # 【Shared Dialogue History】:
        # {chat_history if chat_history else "(None yet)"}

        # Please analyze the available clues, communicate if needed, and try to deduce the murderer.
        # IMPORTANT: Do not ask more than **two questions** in a round. Limit question loops.
        # IMPORTANT: If you are asked a question in previous round, you may respond and optionally ask ONE follow-up.
        # Remember to include:
        # I believe the murderer is: XXX
        # """.strip()

        final_prompt = f"""
        {formatted_prompt}

        【Shared Dialogue History】:
        {chat_history if chat_history else "(None yet)"}
        """.strip()

        return final_prompt

    # def extract_suspect_from(self, response: str) -> str | None:
    #     for line in response.splitlines():
    #         if line.strip().startswith("I believe the murderer is:"):
    #             return line.split(":", 1)[1].strip()
    #     return None
    
    def extract_suspect_from(self, response: str) -> str | None:
        pattern = r"I believe the murderer is[:：]\s*(.+)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def save_log(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"🕵️‍♂️ Multi-Agent Detective Dialogue Log ({timestamp})\n")
            f.write("=" * 50 + "\n\n")
            for line in self.log_lines:
                f.write(line + "\n\n")
        print(f"\n📝 Dialogue log saved to: {self.log_file}")

    def run_dialogue(self):
        for turn in range(self.max_turns):
            round_header = f"\n🔄 Round {turn + 1}\n" + "-" * 30
            print(round_header)
            self.log_lines.append(round_header.strip())

            responses = {}
            # for agent_name in ["Holmes", "Poirot", "Marple"]: 
            agent_order = ["Holmes", "Poirot", "Marple"] # random
            random.shuffle(agent_order) #
            for agent_name in agent_order: #
                agent = self.agents[agent_name]
                prompt = self.format_input_for_agent(agent_name)
                response = agent.run(prompt)
                self.memory[agent_name].append(response)
                responses[agent_name] = response

                log_entry = f"\n🕵️‍♂️ {agent_name} says:\n{response}"
                print(log_entry)
                self.log_lines.append(f"{agent_name}:\n{response}")
                time.sleep(self.delay)

            suspects = [
                self.extract_suspect_from(responses["Holmes"]),
                self.extract_suspect_from(responses["Poirot"]),
                self.extract_suspect_from(responses["Marple"]),
            ]

            if all(suspects) and len(set(suspects)) == 1:
                consensus = suspects[0]
                if consensus.lower() not in ["unknown", "undetermined", "not sure", "uncertain"]:
                    summary = "\n✅ All three detectives have reached a consensus.\n" + f"🎯 The murderer is: {consensus}"
                    print(summary)
                    self.log_lines.append(summary.strip())
                    break

        else:
            conclusion = "\n❌ Reached maximum rounds without consensus among the detectives."
            print(conclusion)
            self.log_lines.append(conclusion.strip())

        self.save_log()
