import os
import csv
import yaml
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("nvidia_key_3b.env")

API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.2-3b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")


# ==========================
# YAML Loading (Unified)
# ==========================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cases(folder="cases", count=3):
    """批量读取 case1.yaml, case2.yaml ..."""
    return [
        load_yaml(os.path.join(folder, f"case{i}.yaml"))["case"]
        for i in range(1, count + 1)
    ]


def load_persona(name, folder="prompts"):
    """加载 persona 并组合成 system prompt"""

    data = load_yaml(os.path.join(folder, f"{name}.yaml"))

    role_play = "\n".join(f"- {item['description']}" for item in data.get("role_play", []))
    protective = "\n".join(f"- {item['description']}" for item in data.get("protective", []))

    return f"""
You are to fully adopt the following persona and behavioral identity.

### Required Persona Traits:
{role_play}

### Forbidden Behaviors:
{protective}

Persona rules override all other instructions.
NEVER break character.
""".strip()


# ==========================
# LLM Call
# ==========================

def ask_killer(case, persona_name):
    system_prompt = load_persona(persona_name)

    user_prompt = f"""
As a detective, determine the most likely killer from the following fictional murder case.
You must stay in your persona throughout this task.
Perform all reasoning internally. Output ONLY the killer's NAME. No explanations. No punctuation. No additional words

Here is the case:

Victim: {case['victim']['name']}
Suspects: {', '.join(s['name'] for s in case['suspects'])}
Cause of Death: {case.get('forensic_evidence', {}).get('cause_of_death', 'N/A')}
Timeline: {'; '.join(case['timeline'])}
Crime Scene: {case['crime_scene'].get('body_state', 'N/A')}

Output format:
KILLER_NAME_HERE
""".strip()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 20
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"].strip()


# ==========================
# Main Logic
# ==========================

def main():

    cases = load_cases()
    personas = ["holmes", "marple", "poirot"]

    # CSV 输出文件
    os.makedirs("evaluation", exist_ok=True)
    csv_filename = f"1.0_wpersona_{datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_path = os.path.join("evaluation", csv_filename)

    # 写入 CSV 表头
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["case", "detective", "believed_murderer"])

        # 每个 persona
        for persona in personas:
            print(f"\n=== Running persona: {persona.upper()} ===")

            # 每个 case
            for idx, case in enumerate(cases, start=1):
                print(f"Case {idx}: running 50 iterations...")
                for i in range(50):
                    killer = ask_killer(case, persona)
                    writer.writerow([idx, persona, killer])

    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
