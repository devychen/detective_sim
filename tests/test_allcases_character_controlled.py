"""
test_allcases_character_conditioned.py

Purpose
-------
This script performs a *character-conditioned single-agent solvability test*
for all murder cases used in the experiment.

Compared to `test_allcases.py`, this script introduces exactly ONE new factor:
a fixed detective persona (e.g. Sherlock Holmes), provided via a system prompt.

What remains unchanged:
- The agent receives the *complete* case information.
- There is only *one* agent.
- There is *no* collaboration or dialogue.
- There is *no* information asymmetry.
- The task is still zero-shot (no examples).

The objective is to verify that:
1. Each case remains solvable under persona constraints.
2. Character role-play alone does not make a solvable case unsolvable.
3. Any failures in later multi-agent settings are not caused by persona conditioning itself.

In the broader research design, this script serves as a controlled intermediate
baseline between:
(1) a universal, non-role-based detective, and
(2) a fully character-driven multi-agent system.
"""


import os
import openai
import yaml
from dotenv import load_dotenv
from datetime import datetime

# Load API key from environment file
load_dotenv('openai_key.env')
# Initialise OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =================================================
# Case Loading
# =================================================

def load_case_files():
    """
    Load and parse all murder case files.

    This function assumes a fixed set of case files stored as:
        cases/case1.yaml
        cases/case2.yaml
        cases/case3.yaml

    Each YAML file is expected to contain a top-level field named 'case',
    which stores all structured information related to that murder case.

    Returns
    -------
    list of dict
        A list of case dictionaries, where each dictionary contains
        the full information for one murder case.
    """
    cases = []
    
    for i in range(1, 4):
        file_path = os.path.join('cases', f'case{i}.yaml')

        with open(file_path, 'r', encoding='utf-8') as file:
            case_data = yaml.safe_load(file)

            # Only the nested 'case' field is used
            cases.append(case_data['case'])

    return cases

# =================================================
# Character Persona Loading
# =================================================

def load_character_prompt(character_name):
    """
    Load a detective persona prompt from a YAML file.

    The persona file is expected to be stored in:
        prompts/{character_name}.yaml

    Only the following sections are used:
    - role_play: descriptions of personality, language style, and reasoning patterns
    - protective: behavioural constraints the agent must obey

    Importantly, the 'task' field (if present) is deliberately ignored.
    This ensures that:
    - the agent's task remains identical to the universal baseline
    - no additional information framing is introduced

    Parameters
    ----------
    character_name : str
        Name of the detective persona (e.g. 'holmes', 'marple', 'poirot')

    Returns
    -------
    str
        A formatted system prompt encoding the character persona.
    """
    file_path = os.path.join('prompts', f'{character_name}.yaml')

    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    role_play = data.get('role_play', [])
    protective = data.get('protective', [])

    persona_prompt_parts = []

    persona_prompt_parts.append(
        "You must strictly adopt the following detective persona and investigative style:"
    )

    # Add role-play descriptions
    for item in role_play:
        persona_prompt_parts.append(f"- {item['description']}")

    persona_prompt_parts.append(
        "\nYou must also obey the following behavioural constraints:"
    )

    # Add protective constraints
    for item in protective:
        persona_prompt_parts.append(f"- {item['description']}")

    return "\n".join(persona_prompt_parts)

# =================================================
# Case Analysis (Single-Agent, Character-Conditioned)
# =================================================

def analyse_case(case_data, case_number, character_name):
    """
    Analyse a single murder case using a character-conditioned LLM.

    The agent:
    - receives the complete case information in one prompt,
    - adopts a fixed detective persona via the system message,
    - does not interact with any other agents,
    - does not receive examples or dialogue history.

    The purpose of this function is not to optimise accuracy,
    but to test whether the case remains solvable under
    persona-based reasoning constraints.

    Parameters
    ----------
    case_data : dict
        Structured information describing the murder case.
    case_number : int
        Index of the case (used for logging and reporting).
    character_name : str
        Name of the detective persona to apply.

    Returns
    -------
    str
        A formatted textual analysis generated by the LLM.
    """

    # Extract a human-readable case name
    case_name = case_data.get('setting', 'Unknown Case').split('\n')[0]

    # -------------------------------------------------
    # User Prompt Construction (Task Definition)
    # -------------------------------------------------
    # This prompt is intentionally identical in structure
    # to the universal detective baseline.
    # The ONLY difference in this script is the system prompt.

    user_prompt = f"""Analyse the following murder case carefully and determine who is the most likely killer.
Consider all evidence, motives, opportunities, and forensic findings. Explain your reasoning step-by-step.

Case: {case_name}
Victim: {case_data['victim']['name']}
Suspects: {', '.join([s['name'] for s in case_data['suspects']])}

Key Evidence:
- Crime Scene: {case_data['crime_scene'].get('body_state', 'N/A')}
- Forensic Findings: {case_data.get('forensic_evidence', {}).get('cause_of_death', 'N/A')}
- Timeline: {'; '.join(case_data['timeline'])}

For each suspect, evaluate:
1. Motive strength
2. Opportunity (alibis, access)
3. Physical evidence linking them
4. Behavioural clues

After analysing all factors, conclude with the most probable killer and a brief explanation.
"""

    # -------------------------------------------------
    # System Prompt Construction (Persona Only)
    # -------------------------------------------------

    system_prompt = load_character_prompt(character_name)

    # -------------------------------------------------
    # LLM Call
    # -------------------------------------------------
    # A low temperature is used to prioritise logical consistency
    # over stylistic creativity.

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return (
        f"Case {case_number} Analysis (Character: {character_name.capitalize()}):\n"
        f"{response.choices[0].message.content}\n"
    )

# =================================================
# Result Storage
# =================================================

def save_results_to_file(results, filename=None):
    """
    Save all case analyses to a timestamped text file.

    The output is intended for:
    - human inspection,
    - qualitative comparison across experimental conditions,
    - inclusion as supplementary research material.

    Parameters
    ----------
    results : str
        Combined analysis text for all cases.
    filename : str or None
        Optional custom filename.

    Returns
    -------
    str
        The path to the saved output file.
    """
    # Ensure output directory exists
    output_dir = "tests"
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"case_analysis_character_conditioned_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("MURDER CASE ANALYSIS REPORT\n")
        file.write(f"Generated on: {datetime.now().strftime('%d %B %Y at %H:%M')}\n")
        file.write("=" * 60 + "\n\n")
        file.write(results)

    return filename

# =================================================
# Main Execution
# =================================================

def main():
    """
    Run character-conditioned solvability analysis for all cases.

    Each case is analysed:
    - independently,
    - by a single agent,
    - with a fixed detective persona,
    - using complete case information.

    The experiment is repeated for multiple personas
    (e.g. Holmes, Marple, Poirot), with results saved separately.
    """

    # Load all case data once
    cases = load_case_files()

    # List of detective personas to evaluate
    character_list = ["holmes", "marple", "poirot"]

    for character_name in character_list:
        print(f"\n{'#' * 60}")
        print(f"Running character-conditioned analysis: {character_name.capitalize()}")
        print(f"{'#' * 60}\n")

        all_results = ""

        for i, case_data in enumerate(cases, 1):
            print(f"\n{'=' * 40}")
            print(f"Analysing Case {i} with {character_name.capitalize()}...")

            case_analysis = analyse_case(case_data, i, character_name)

            print(case_analysis)
            all_results += case_analysis + "\n" + "=" * 40 + "\n\n"

        # Generate timestamp for this character's run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results to a character- and time-specific file
        output_file = save_results_to_file(
            all_results,
            filename=f"case_analysis_{character_name}_{timestamp}.txt"
        )

        print(f"\nResults for {character_name.capitalize()} saved to: {output_file}")

# =================================================
# Script Entry Point
# =================================================

if __name__ == "__main__":
    main()
