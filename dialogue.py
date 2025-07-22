import time
from datetime import datetime
import yaml
import os

from agents.holmes_agent import create_holmes_agent
from agents.poirot_agent import create_poirot_agent
from agents.marple_agent import create_marple_agent

class DetectiveDialogue:
    def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt", clue_file="tasks/test_clues.yaml"):
        self.max_turns = max_turns
        self.delay = delay
        self.log_file = log_file
        self.log_lines = []

        # Load clues from YAML file
        if not os.path.exists(clue_file):
            raise FileNotFoundError(f"Clue file not found: {clue_file}")

        with open(clue_file, 'r', encoding='utf-8') as f:
            self.clues = yaml.safe_load(f)

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
                history_parts.append(f"【{name}】said:\n" + "\n".join(messages))

        chat_history = "\n\n".join(history_parts)
        clue = self.clues.get(agent_name, "")

        prompt = f"""
            You are {agent_name}. Here is the current shared information:

            {chat_history if chat_history else "No conversation yet."}

            Your exclusive clue is: {clue}

            Please analyze the available clues and deduce who the murderer might be, explaining your reasoning.
            If needed, you can address direct questions to other detectives, but avoid repeating the same questions.
            Unless you have a different perspective, don't repeat your statements.
            Be concise.
            Please strictly use the following format for your conclusion:
            I believe the murderer is: XXX
                    """.strip()

        return prompt

    def extract_suspect_from(self, response: str) -> str | None:
        for line in response.splitlines():
            if line.strip().startswith("I believe the murderer is:"):
                return line.split(":", 1)[1].strip()
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
            for agent_name in ["Holmes", "Poirot", "Marple"]:
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
                summary = "\n✅ All three detectives have reached a consensus.\n" + f"🎯 The murderer is: {suspects[0]}"
                print(summary)
                self.log_lines.append(summary.strip())
                break
        else:
            conclusion = "\n❌ Reached maximum rounds without consensus among the detectives."
            print(conclusion)
            self.log_lines.append(conclusion.strip())

        self.save_log()