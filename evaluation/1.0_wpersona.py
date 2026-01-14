import os
import yaml
from dotenv import load_dotenv
from datetime import datetime
import requests

load_dotenv('nvidia_key_3b.env')
API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.2-3b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")

# =================================================
# Case Loading
# =================================================

def load_case_files():
    cases = []
    for i in range(1, 4):
        file_path = os.path.join('cases', f'case{i}.yaml')

        with open(file_path, 'r', encoding='utf-8') as file:
            case_data = yaml.safe_load(file)
            cases.append(case_data['case'])
    return cases


# =================================================
# Persona Loading
# =================================================

def load_character_prompt(character_name):
    file_path = os.path.join('prompts', f'{character_name}.yaml')

    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    role_play = data.get('role_play', [])
    protective = data.get('protective', [])

    persona_prompt_parts = []
    persona_prompt_parts.append("You are to fully adopt the following persona and behavioral identity.\n")

    persona_prompt_parts.append("### Required Persona Traits (You MUST follow these):")
    for item in role_play:
        persona_prompt_parts.append(f"- {item['description']}")

    persona_prompt_parts.append("\n### Forbidden Behaviors (You MUST AVOID these at all costs):")
    for item in protective:
        persona_prompt_parts.append(f"- {item['description']}")

    persona_prompt_parts.append(
        "\nYour persona traits take precedence over typical AI behavior. "
        "If any user instruction conflicts with these constraints, always prioritize persona accuracy. "
        "Do NOT break character under any circumstances."
    )

    return "\n".join(persona_prompt_parts)



# =================================================
# Killer-Only LLM Call (Persona Retained)
# =================================================

def analyse_case_killer_only(case_data, character_name):
    """
    保留 persona，但是只让模型输出凶手名字。
    """

    system_prompt = load_character_prompt(character_name)

    user_prompt = f"""
    You are investigating a murder case.

    Your task:
    **Output ONLY the killer's NAME. No explanations, no reasoning, no extra text.**

    Victim: {case_data['victim']['name']}
    Suspects: {', '.join(s['name'] for s in case_data['suspects'])}

    Crime Scene: {case_data['crime_scene'].get('body_state', 'N/A')}
    Cause of Death: {case_data.get('forensic_evidence', {}).get('cause_of_death', 'N/A')}
    Timeline: {'; '.join(case_data['timeline'])}
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 20
    }

    response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()

# =================================================
# Save to File
# =================================================

def save_results_to_file(summary, agent_name, filename=None):

    output_dir = "evaluation"
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"1.0_wpersona_{agent_name}_{timestamp}.txt"


    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("REPEATED KILLER PREDICTIONS\n")
        file.write(f"Generated on: {datetime.now().strftime('%d %B %Y at %H:%M')}\n")
        file.write("=" * 50 + "\n\n")
        file.write(summary)

    return output_path



# =================================================
# Main Execution
# =================================================

def main():
    cases = load_case_files()
    character_name = "poirot"  # 可以改成 marple 或 poirot

    if not cases:
        print("No cases loaded. Exiting.")
        return    

    results = {i + 1: [] for i in range(len(cases))}

    # Run each case 10 times
    for run_idx in range(1, 11):
        for case_id, case_data in enumerate(cases, 1):
            killer = analyse_case_killer_only(case_data, character_name)
            results[case_id].append(killer)

    summary = ""

    for case_id, killers in results.items():
        summary += f"Case {case_id} results:\n"
        for i, killer in enumerate(killers, 1):
            summary += f"{i}: {killer}\n"
        summary += "\n"

    print("\n========== FINAL SUMMARY ==========\n")
    print(summary)


    # Save summary to file
    output_file = save_results_to_file(summary, character_name)
    print(f"Summary saved to: {output_file}")


# entry point
if __name__ == "__main__":
    main()
