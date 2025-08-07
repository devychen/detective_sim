# main_debate.py
# PURPOSE: to run the debate and save output as CSV

import random
import csv
from pathlib import Path
import yaml

from agents.holmes_agent import create_holmes_agent
from agents.poirot_agent import create_poirot_agent
from agents.marple_agent import create_marple_agent

from debate import build_system_prompt

NUM_ROUNDS = 10
PROMPT_TEMPLATE_PATH = Path("rules/rule_debate.yaml")

def initialise_agents():
    agents = {
        "Holmes": create_holmes_agent(),
        "Poirot": create_poirot_agent(),
        "Marple": create_marple_agent()
    }
    return agents

def inject_system_prompt(agent, system_prompt: str):
    """Inject system prompt into memory as initial context (optional)"""
    agent.update_memory("System", system_prompt)

def main():
    agents = initialise_agents()
    system_prompts = {}  # 用于收集所有agent的system prompt

    # Build and inject system prompts
    for name, agent in agents.items():
        sys_prompt = build_system_prompt(name, PROMPT_TEMPLATE_PATH)
        inject_system_prompt(agent, sys_prompt)
        system_prompts[name] = sys_prompt  # 收集system prompt

    # 保存system prompts到yaml文件
    system_prompt_path = Path("data/debate_system_prompts.yaml")
    system_prompt_path.parent.mkdir(exist_ok=True)
    with open(system_prompt_path, "w", encoding="utf-8") as f:
        yaml.dump(system_prompts, f, allow_unicode=True, default_flow_style=False)

    # 开始辩论流程（与原来一致）
    csv_rows = []
    turn_number = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        speaking_order = random.sample(list(agents.keys()), k=len(agents))

        for name in speaking_order:
            agent = agents[name]
            input_text = "Please continue your argument or respond to others."
            response = agent.run(input_text).strip()

            # Append row to CSV data
            csv_rows.append([f"Turn {turn_number}", name, response])
            turn_number += 1

            # Update memory for all other agents
            for other_name, other_agent in agents.items():
                if other_name != name:
                    other_agent.update_memory(name, response)

    # Save conversation log to CSV
    outpath = Path("data/debate_run_001.csv")
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Turn", "Agent", "Content"])
        writer.writerows(csv_rows)

if __name__ == "__main__":
    main()


# ========================================旧的，没有csv，没有print prompt

# # main_debate.py
# # PURPOSE: to run the debate

# import random
# import json
# from pathlib import Path

# from agents.holmes_agent import create_holmes_agent
# from agents.poirot_agent import create_poirot_agent
# from agents.marple_agent import create_marple_agent

# from debate import build_system_prompt

# NUM_ROUNDS = 10
# PROMPT_TEMPLATE_PATH = Path("rules/rule_debate.yaml")

# def initialize_agents():
#     agents = {
#         "Holmes": create_holmes_agent(),
#         "Poirot": create_poirot_agent(),
#         "Marple": create_marple_agent()
#     }
#     return agents

# def inject_system_prompt(agent, system_prompt: str):
#     """Inject system prompt into memory as initial context (optional)"""
#     agent.update_memory("System", system_prompt)

# def main():
#     agents = initialize_agents()

#     # Build and inject system prompts
#     for name, agent in agents.items():
#         sys_prompt = build_system_prompt(name, PROMPT_TEMPLATE_PATH)
#         inject_system_prompt(agent, sys_prompt)

#     chat_log = []

#     for round_num in range(1, NUM_ROUNDS + 1):
#         print(f"\n=== Round {round_num} ===")
#         speaking_order = random.sample(list(agents.keys()), k=len(agents))

#         for name in speaking_order:
#             agent = agents[name]
#             input_text = "Please continue your argument or respond to others."
#             response = agent.run(input_text).strip()

#             print(f"\n{name}: {response}")
#             chat_log.append({
#                 "round": round_num,
#                 "speaker": name,
#                 "text": response,
#                 "order": speaking_order
#             })

#             # Update memory for all other agents
#             for other_name, other_agent in agents.items():
#                 if other_name != name:
#                     other_agent.update_memory(name, response)

#     # Save conversation log
#     outpath = Path("data/debate_run_001.json")
#     outpath.parent.mkdir(exist_ok=True)
#     with open(outpath, "w", encoding="utf-8") as f:
#         json.dump(chat_log, f, indent=2, ensure_ascii=False)

#     print(f"\nDebate complete. Log saved to {outpath}")

# if __name__ == "__main__":
#     main()
