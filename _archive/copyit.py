        # tell models to generate direct questions to address, each other.
        # give them an initial setting
        # do not repeat questions
        # do not repeat statements unless being disagree
        # "be concise" instead of max_length
        # repeating is a common issue. 1/ stop early. 2/ change the clues. (mixed) 3/ shuffle agents in each turns.
        # 3 cases x 10 simulations x 10 turns max
        # 1 debate x 10 simuations x 10 turns max

# import time
# from datetime import datetime
# import yaml
# import os

# from agents.holmes_agent import create_holmes_agent
# from agents.poirot_agent import create_poirot_agent
# from agents.marple_agent import create_marple_agent

# class DetectiveDialogue:
#     def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt", clue_file="tasks/test_clues.yaml"):
#         self.max_turns = max_turns
#         self.delay = delay
#         self.log_file = log_file
#         self.log_lines = []

#         # 从 YAML 文件加载线索
#         if not os.path.exists(clue_file):
#             raise FileNotFoundError(f"Clue File Not Found：{clue_file}")

#         with open(clue_file, 'r', encoding='utf-8') as f:
#             self.clues = yaml.safe_load(f)

#         # 初始化 memory
#         self.memory = {
#             "Holmes": [],
#             "Poirot": [],
#             "Marple": [],
#         }

#         # 初始化 agents
#         self.holmes = create_holmes_agent()
#         self.poirot = create_poirot_agent()
#         self.marple = create_marple_agent()


#         self.agents = {
#             "Holmes": self.holmes,
#             "Poirot": self.poirot,
#             "Marple": self.marple,
#         }

#     def format_input_for_agent(self, agent_name: str) -> str:
#         history_parts = []
#         for name, messages in self.memory.items():
#             if messages:
#                 history_parts.append(f"【{name}】Said：\n" + "\n".join(messages))

#         chat_history = "\n\n".join(history_parts)
#         clue = self.clues.get(agent_name, "")

#         prompt = f"""
#             You are {agent_name}。Below is the currently shared information:

#             {chat_history if chat_history else "No conversation yet。"}

#             Your exclusive clue is：{clue}

#             Analyze the given clues and deduce the most likely suspect. Explain your reasoning.
#             Strictly follow this format for your conclusion:
#             I think the murderer is: XXX
#                     """.strip()
#         # tell models to generate direct questions to address, each other.
#         # give them an initial setting
#         # do not repeat questions
#         # do not repeat statements unless being disagree
#         # "be concise" instead of max_length
#         # repeating is a common issue. 1/ stop early. 2/ change the clues. (mixed) 3/ shuffle agents in each turns.
#         # 3 cases x 10 simulations x 10 turns max
#         # 1 debate x 10 simuations x 10 turns max

#         return prompt

#     def extract_suspect_from(self, response: str) -> str | None:
#         for line in response.splitlines():
#             if line.strip().startswith("I think the killer is："):
#                 return line.split("：", 1)[1].strip()
#         return None

#     def save_log(self):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(self.log_file, "w", encoding="utf-8") as f:
#             f.write(f"🕵️‍♂️ Multi-Detective Agent Chat History（{timestamp}）\n")
#             f.write("=" * 50 + "\n\n")
#             for line in self.log_lines:
#                 f.write(line + "\n\n")
#         print(f"\n📝 Dialogue saved to：{self.log_file}")

#     def run_dialogue(self):
#         for turn in range(self.max_turns):
#             round_header = f"\n🔄 Round {turn + 1} \n" + "-" * 30
#             print(round_header)
#             self.log_lines.append(round_header.strip())

#             responses = {}
#             for agent_name in ["Holmes", "Poirot", "Marple"]:
#                 agent = self.agents[agent_name]
#                 prompt = self.format_input_for_agent(agent_name)
#                 response = agent.run(prompt)
#                 self.memory[agent_name].append(response)
#                 responses[agent_name] = response

#                 log_entry = f"\n🕵️‍♂️ {agent_name} 说：\n{response}"
#                 print(log_entry)
#                 self.log_lines.append(f"{agent_name}:\n{response}")
#                 time.sleep(self.delay)

#             suspects = [
#                 self.extract_suspect_from(responses["Holmes"]),
#                 self.extract_suspect_from(responses["Poirot"]),
#                 self.extract_suspect_from(responses["Marple"]),
#             ]

#             if all(suspects) and len(set(suspects)) == 1:
#                 summary = "\n✅ Mutual Agreement Reached。\n" + f"🎯 The killer is：{suspects[0]}"
#                 print(summary)
#                 self.log_lines.append(summary.strip())
#                 break
#         else:
#             conclusion = "\n❌ Reached maximum turns, no mutual agreement reached。"
#             print(conclusion)
#             self.log_lines.append(conclusion.strip())

#         self.save_log()


# # # dialogue.py

# # import time
# # from datetime import datetime

# # from agents.holmes_agent import create_holmes_agent
# # from agents.poirot_agent import create_poirot_agent
# # from agents.marple_agent import create_marple_agent


# # class DetectiveDialogue:
# #     def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt"):
# #         self.max_turns = max_turns
# #         self.delay = delay
# #         self.log_file = log_file
# #         self.log_lines = []

# #         # 加载线索
# #         with open(clue_file, 'r', encoding='utf-8') as f:
# #             self.clues = yaml.safe_load(f)

# #         self.memory = {
# #             "Holmes": [],
# #             "Poirot": [],
# #             "Marple": [],
# #         }

# #         # 初始化 agents，并提供提问接口
# #         self.holmes = create_holmes_agent(self.ask_another_agent)
# #         self.poirot = create_poirot_agent(self.ask_another_agent)
# #         self.marple = create_marple_agent(self.ask_another_agent)

# #         self.agents = {
# #             "Holmes": self.holmes,
# #             "Poirot": self.poirot,
# #             "Marple": self.marple,
# #         }

# #     def format_input_for_agent(self, agent_name: str) -> str:
# #         history_parts = []
# #         for name, messages in self.memory.items():
# #             if messages:
# #                 history_parts.append(f"【{name}】说：\n" + "\n".join(messages))

# #         chat_history = "\n\n".join(history_parts)
# #         clue = self.clues.get(agent_name, "")

# #         prompt = f"""
# #             你是{agent_name}。以下是目前共享的信息：

# #             {chat_history if chat_history else "目前尚无对话。"}

# #             你掌握的独家线索是：{clue}

# #             请根据已有线索进行分析，推测可能的凶手是谁，并说明理由。你可以调用工具 ask_other_agent 来提问另一位侦探。
# #             请严格使用以下格式写出你的结论：
# #             我认为凶手是：XXX
# #                     """.strip()

# #         return prompt

# #     def extract_suspect_from(self, response: str) -> str | None:
# #         for line in response.splitlines():
# #             if line.strip().startswith("我认为凶手是："):
# #                 return line.split("：", 1)[1].strip()
# #         return None

# #     def ask_another_agent(self, target_agent: str, question: str) -> str:
# #         if target_agent not in self.agents:
# #             return f"[Error] 没有名为 {target_agent} 的侦探。"
# #         print(f"\n🔧 工具调用：向 {target_agent} 提问：{question}")
# #         response = self.agents[target_agent].run(question)
# #         print(f"🔧 {target_agent} 回答：{response}")
# #         return response

# #     def save_log(self):
# #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #         with open(self.log_file, "w", encoding="utf-8") as f:
# #             f.write(f"🕵️‍♂️ 多智能体探案对话记录（{timestamp}）\n")
# #             f.write("=" * 50 + "\n\n")
# #             for line in self.log_lines:
# #                 f.write(line + "\n\n")
# #         print(f"\n📝 对话日志已保存至：{self.log_file}")

# #     def run_dialogue(self):
# #         for turn in range(self.max_turns):
# #             round_header = f"\n🔄 第 {turn + 1} 轮对话\n" + "-" * 30
# #             print(round_header)
# #             self.log_lines.append(round_header.strip())

# #             responses = {}
# #             for agent_name in ["Holmes", "Poirot", "Marple"]:
# #                 agent = self.agents[agent_name]
# #                 prompt = self.format_input_for_agent(agent_name)
# #                 response = agent.run(prompt)
# #                 self.memory[agent_name].append(response)
# #                 responses[agent_name] = response

# #                 log_entry = f"\n🕵️‍♂️ {agent_name} 说：\n{response}"
# #                 print(log_entry)
# #                 self.log_lines.append(f"{agent_name}:\n{response}")
# #                 time.sleep(self.delay)

# #             suspects = [
# #                 self.extract_suspect_from(responses["Holmes"]),
# #                 self.extract_suspect_from(responses["Poirot"]),
# #                 self.extract_suspect_from(responses["Marple"]),
# #             ]

# #             if all(suspects) and len(set(suspects)) == 1:
# #                 summary = "\n✅ 三位侦探已达成一致意见。\n" + f"🎯 凶手是：{suspects[0]}"
# #                 print(summary)
# #                 self.log_lines.append(summary.strip())
# #                 break
# #         else:
# #             conclusion = "\n❌ 达到最大轮数，三位侦探仍未达成一致结论。"
# #             print(conclusion)
# #             self.log_lines.append(conclusion.strip())

# #         self.save_log()



# # import time
# # from datetime import datetime

# # from agents.holmes_agent import create_holmes_agent
# # from agents.poirot_agent import create_poirot_agent
# # from agents.marple_agent import create_marple_agent

# # class DetectiveDialogue:
# #     def __init__(self, max_turns=10, delay=1, log_file="dialogue_log.txt"):
# #         self.holmes = create_holmes_agent()
# #         self.poirot = create_poirot_agent()
# #         self.marple = create_marple_agent()
# #         self.max_turns = max_turns
# #         self.delay = delay
# #         self.log_file = log_file

# #         self.clues = {
# #             "Holmes": "你在案发现场。你观察到了尸体的状态、周围环境和潜在物证。",
# #             "Poirot": "你了解案发当天发生的事件、时间线和人物动向。",
# #             "Marple": "你调查了所有嫌疑人的背景、动机和人际关系网络。",
# #         }

# #         self.log_lines = []

# #     def make_input(self, agent_name):
# #         clue = self.clues.get(agent_name, "")
# #         return f"""你是{agent_name}。
# # 你掌握的独家线索是：{clue}
# # 请你根据已有线索进行分析，推测可能的凶手是谁，并说明理由。
# # 请严格使用以下格式写出你的结论：
# # 我认为凶手是：XXX"""

# #     def extract_suspect_from(self, response):
# #         for line in response.splitlines():
# #             line = line.strip()
# #             if line.startswith("我认为凶手是："):
# #                 return line.split("：", 1)[1].strip()
# #         return None

# #     def save_log(self):
# #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #         with open(self.log_file, "w", encoding="utf-8") as f:
# #             f.write(f"🕵️‍♂️ 多智能体探案对话记录（{timestamp}）\n")
# #             f.write("=" * 50 + "\n\n")
# #             for line in self.log_lines:
# #                 f.write(line + "\n\n")
# #         print(f"\n📝 对话日志已保存至：{self.log_file}")

# #     def run_dialogue(self):
# #         for turn in range(self.max_turns):
# #             round_header = f"\n🔄 第 {turn + 1} 轮对话\n" + "-" * 30
# #             print(round_header)
# #             self.log_lines.append(round_header.strip())

# #             responses = {}
# #             for agent_name, agent in [
# #                 ("Holmes", self.holmes),
# #                 ("Poirot", self.poirot),
# #                 ("Marple", self.marple),
# #             ]:
# #                 prompt = self.make_input(agent_name)
# #                 response = agent.run(prompt)
# #                 responses[agent_name] = response

# #                 log_entry = f"\n🕵️‍♂️ {agent_name} 说：\n{response}"
# #                 print(log_entry)
# #                 self.log_lines.append(f"{agent_name}:\n{response}")
# #                 time.sleep(self.delay)

# #             suspects = [
# #                 self.extract_suspect_from(responses["Holmes"]),
# #                 self.extract_suspect_from(responses["Poirot"]),
# #                 self.extract_suspect_from(responses["Marple"]),
# #             ]

# #             if all(suspects) and len(set(suspects)) == 1:
# #                 summary = "\n✅ 三位侦探已达成一致意见。\n" + f"🎯 凶手是：{suspects[0]}"
# #                 print(summary)
# #                 self.log_lines.append(summary.strip())
# #                 break
# #         else:
# #             conclusion = "\n❌ 达到最大轮数，三位侦探仍未达成一致结论。"
# #             print(conclusion)
# #             self.log_lines.append(conclusion.strip())

# #         self.save_log()
