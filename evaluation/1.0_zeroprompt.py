"""
1.0_zeroprompt.py

本版本使用llama。gpt版本见test > test_allcases.py
这个版本直接输出结果，不包含分析的内容
Runs each case 10 times and records the killer prediction of each run.


Purpose
-------
This script performs a *zero-shot solvability verification* for all murder cases
used in the multi-agent experiment.

"""

import os
from dotenv import load_dotenv
import yaml
from datetime import datetime
import requests

# Load NVIDIA API key
load_dotenv('nvidia_key_3b.env')
API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.2-3b-instruct"
API_KEY = os.getenv("NVIDIA_API_KEY")


# =================================================
# Case Loader (Corrected)
# =================================================

def load_case_files():
    """
    Load case files based on environment variable RUN_CASES.
    Returns dict: {case_id: case_data}

    Examples:
        RUN_CASES not set  -> loads case1, case2, case3
        RUN_CASES=3        -> loads only case3
        RUN_CASES=1,3      -> loads case1 and case3
    """

    run_cases_env = os.getenv("RUN_CASES")

    if run_cases_env:
        case_ids = [int(x.strip()) for x in run_cases_env.split(",")]
    else:
        case_ids = [1, 2, 3]

    cases = {}
    for i in case_ids:
        file_path = os.path.join('cases', f'case{i}.yaml')
        if not os.path.exists(file_path):
            print(f"WARNING: case file {file_path} does not exist. Skipping.")
            continue

        with open(file_path, 'r', encoding='utf-8') as file:
            case_yaml = yaml.safe_load(file)
            cases[i] = case_yaml['case']

    return cases


# =================================================
# LLM Call - Killer Only
# =================================================

def analyse_case(case_data):
    """
    Ask LLM to output ONLY the killer name.
    """

    prompt = f"""
    Determine the most likely killer from the following murder case.
    Output ONLY the killer's NAME. No explanations.

    Victim: {case_data['victim']['name']}
    Suspects: {', '.join([s['name'] for s in case_data['suspects']])}

    Crime Scene: {case_data['crime_scene'].get('body_state', 'N/A')}
    Cause of Death: {case_data.get('forensic_evidence', {}).get('cause_of_death', 'N/A')}
    Timeline: {'; '.join(case_data['timeline'])}
    """

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 30
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" not in data:
        raise RuntimeError(f"NVIDIA API request failed: {data}")

    return data["choices"][0]["message"]["content"].strip()


# =================================================
# Save to File
# =================================================

def save_results_to_file(summary, filename=None):

    output_dir = "evaluation"
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"1.0_zeroprompt_{timestamp}.txt"

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

    if not cases:
        print("No cases loaded. Exiting.")
        return

    # Initialize results container using actual case IDs
    results = {case_id: [] for case_id in cases}

    # Run each loaded case 10 times (silent mode)
    for run_idx in range(1, 11):
        for case_id, case_data in cases.items():
            killer = analyse_case(case_data)
            results[case_id].append(killer)

    # Build summary output
    summary = ""

    for case_id in sorted(results.keys()):
        summary += f"Case {case_id} results:\n"
        for run_idx, killer in enumerate(results[case_id], 1):
            summary += f"{run_idx}: {killer}\n"
        summary += "\n"

    # Display ONLY final summary
    print("\n========== FINAL SUMMARY ==========\n")
    print(summary)

    # Save summary to file
    output_file = save_results_to_file(summary)
    print(f"Summary saved to: {output_file}")


# Entry point
if __name__ == "__main__":
    main()
