# collab.py
import yaml
from agents.holmes_agent import create_holmes_agent
from agents.poirot_agent import create_poirot_agent
from agents.marple_agent import create_marple_agent

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_prompt(agent_name: str, agent_config: dict, rules: dict, case_data: dict, chat_history: str) -> str:
    """组装 prompt"""

    # ===== 全局规则 =====
    common_intro = rules.get("common_intro", "").format(agent_name=agent_name)
    common_rules = rules.get("common_rules", "")

    # ===== agent 专属 prompt =====
    role_play_parts = "\n".join(
        [f"- {item['description']}\nExample: {item['example']}" for item in agent_config.get("role_play", [])]
    )
    protective_parts = "\n".join(
        [f"- {item['description']}" for item in agent_config.get("protective", [])]
    )
    task = agent_config.get("task", "")

    # ===== 案件数据 =====
    case_info = []
    if "setting" in case_data:
        case_info.append(f"Setting:\n{case_data['setting']}")
    if "victim" in case_data:
        case_info.append(f"Victim:\n{case_data['victim']}")
    if "suspects" in case_data:
        case_info.append("Suspects:")
        for s in case_data["suspects"]:
            case_info.append(f"- {s}")
    if "forensic_evidence" in case_data:
        case_info.append(f"Forensic Evidence:\n{case_data['forensic_evidence']}")

    case_text = "\n".join(case_info)

    # ===== 拼接最终 prompt =====
    prompt = f"""
{common_intro}

{common_rules}

=== Role Play Guidelines for {agent_name} ===
{role_play_parts}

=== Protective Guidelines ===
{protective_parts}

=== Task ===
{task}

=== Case Information ===
{case_text}

=== Conversation So Far ===
{chat_history}

Now, please respond in character as {agent_name}.
"""
    return prompt.strip()

def format_input_for_agent(agent_name: str, agent_yaml_path: str, rules_path: str, case_path: str, chat_history: str):
    agent_config = load_yaml(agent_yaml_path)
    rules = load_yaml(rules_path)
    case_data = load_yaml(case_path)
    return build_prompt(agent_name, agent_config, rules, case_data, chat_history)

def run_collab():
    # 加载三个 agent
    holmes = create_holmes_agent()
    poirot = create_poirot_agent()
    marple = create_marple_agent()

    # 加载配置
    chat_history = ""
    holmes_prompt = format_input_for_agent("Sherlock Holmes", "prompts/holmes.yaml", "rules/rule_collab.yaml", "cases/case1.yaml", chat_history)
    poirot_prompt = format_input_for_agent("Hercule Poirot", "prompts/poirot.yaml", "rules/rule_collab.yaml", "cases/case1.yaml", chat_history)
    marple_prompt = format_input_for_agent("Miss Marple", "prompts/marple.yaml", "rules/rule_collab.yaml", "cases/case1.yaml", chat_history)

    # 模拟第一轮
    holmes_reply = holmes.run(holmes_prompt)
    chat_history += f"\nHolmes: {holmes_reply}"

    poirot_reply = poirot.run(poirot_prompt + chat_history)
    chat_history += f"\nPoirot: {poirot_reply}"

    marple_reply = marple.run(marple_prompt + chat_history)
    chat_history += f"\nMarple: {marple_reply}"

    print("=== Conversation ===")
    print(chat_history)

if __name__ == "__main__":
    run_collab()
