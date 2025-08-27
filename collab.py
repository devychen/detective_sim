# collab.py
import yaml
from agent.holmes_agent import HolmesAgent
from agent.poirot_agent import PoirotAgent
from agent.marple_agent import MarpleAgent

class DetectiveDialogue:
    def __init__(self, rule_path="rule_collab.yaml", turns=10):
        self.rule_path = rule_path
        self.rules = self._load_rules()
        self.turns = turns

        # 初始化三个探员
        self.agents = [
            HolmesAgent(),
            PoirotAgent(),
            MarpleAgent()
        ]

        # 给每个 agent 设置开场系统提示
        for agent in self.agents:
            agent.update_memory("system", self.rules["common_intro"].format(agent_name=agent.name))
            agent.update_memory("system", self.rules["common_rules"])

    def _load_rules(self):
        with open(self.rule_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def broadcast(self, speaker, content):
        """把某个 agent 的发言广播到其他人"""
        for agent in self.agents:
            if agent.name != speaker.name:
                agent.update_memory(speaker.name, content)

    def run_dialogue(self):
        for turn in range(self.turns):
            print(f"\n--- Round {turn+1} ---")
            for agent in self.agents:
                response = agent.run("Continue the collaborative investigation.")
                print(f"{agent.name}: {response}\n")
                self.broadcast(agent, response)

        print("\n=== Dialogue Finished ===")
