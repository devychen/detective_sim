import time
from datetime import datetime

from agents.holmes_agent import create_holmes_agent
from agents.poirot_agent import create_poirot_agent
from agents.marple_agent import create_marple_agent

class DetectiveDialogue:
    def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt"):
        self.holmes = create_holmes_agent()
        self.poirot = create_poirot_agent()
        self.marple = create_marple_agent()
        self.max_turns = max_turns
        self.delay = delay
        self.log_file = log_file

        # 初始线索
        self.clues = {
            "Holmes": "你在案发现场。你观察到了尸体的状态、周围环境和潜在物证。",
            "Poirot": "你了解案发当天发生的事件、时间线和人物动向。",
            "Marple": "你调查了所有嫌疑人的背景、动机和人际关系网络。",
        }

        self.memory = {
            "Holmes": [],
            "Poirot": [],
            "Marple": [],
        }

        self.log_lines = []

    def format_prompt(self, agent_name):
        full_history = []
        for speaker, messages in self.memory.items():
            if messages:
                full_history.append(f"【{speaker}】说：\n" + "\n".join(messages))
        joined_history = "\n\n".join(full_history)

        clue = self.clues.get(agent_name, "")
        prompt = f"""
你是{agent_name}。以下是目前共享的信息：

{joined_history if joined_history else "目前尚无对话。"}

你掌握的独家线索是：{clue}

请你根据已有线索进行分析，推测可能的凶手是谁，并说明理由。
请严格使用以下格式写出你的结论：
我认为凶手是：XXX
""".strip()

        return prompt

    def extract_suspect_from(self, response):
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("我认为凶手是："):
                return line.split("：", 1)[1].strip()
        return None

    def save_log(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"🕵️‍♂️ 多智能体探案对话记录（{timestamp}）\n")
            f.write("=" * 50 + "\n\n")
            for line in self.log_lines:
                f.write(line + "\n\n")
        print(f"\n📝 对话日志已保存至：{self.log_file}")

    def run_dialogue(self):
        for turn in range(self.max_turns):
            round_header = f"\n🔄 第 {turn + 1} 轮对话\n" + "-" * 30
            print(round_header)
            self.log_lines.append(round_header.strip())

            responses = {}
            for agent_name, agent in [
                ("Holmes", self.holmes),
                ("Poirot", self.poirot),
                ("Marple", self.marple),
            ]:
                prompt = self.format_prompt(agent_name)
                response = agent.run(prompt)
                self.memory[agent_name].append(response)
                responses[agent_name] = response

                log_entry = f"\n🕵️‍♂️ {agent_name} 说：\n{response}"
                print(log_entry)
                self.log_lines.append(f"{agent_name}:\n{response}")
                time.sleep(self.delay)

            suspects = [
                self.extract_suspect_from(responses["Holmes"]),
                self.extract_suspect_from(responses["Poirot"]),
                self.extract_suspect_from(responses["Marple"]),
            ]

            if all(suspects) and len(set(suspects)) == 1:
                summary = "\n✅ 三位侦探已达成一致意见。\n" + f"🎯 凶手是：{suspects[0]}"
                print(summary)
                self.log_lines.append(summary.strip())
                break
        else:
            conclusion = "\n❌ 达到最大轮数，三位侦探仍未达成一致结论。"
            print(conclusion)
            self.log_lines.append(conclusion.strip())

        self.save_log()


if __name__ == "__main__":
    dialogue = DetectiveDialogue()
    dialogue.run_dialogue()
