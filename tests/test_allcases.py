"""
test_allcases.py

Purpose
-------
This script performs a *zero-shot solvability verification* for all murder cases
used in the multi-agent experiment.

Specifically, it uses a single, non-role-based Large Language Model (LLM)
to analyse each case independently, without:
- character role-play,
- multi-agent interaction,
- dialogue history,
- information asymmetry.

The goal is NOT to optimise performance, but to verify that:
1. Each case is internally coherent.
2. The provided evidence is sufficient to support a reasoned inference.
3. A competent, general-purpose LLM can arrive at a plausible conclusion
   using only the case materials.

In the broader research design, this script serves as a *baseline sanity check*
and supports the claim that failures in the multi-agent setting are not caused
by ill-posed or unsolvable cases.
"""

import os
import openai
from dotenv import load_dotenv
import yaml
from datetime import datetime

# Set OpenAI API key
load_dotenv('openai_key.env')
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_case_files():
    """Load and parse case YAML files from the cases directory"""
    cases = []
    for i in range(1, 4):
        file_path = os.path.join('cases', f'case{i}.yaml')
        with open(file_path, 'r', encoding='utf-8') as file:
            case_data = yaml.safe_load(file)
            cases.append(case_data['case'])  # Access the nested 'case' data
    return cases

def analyse_case(case_data, case_number):  
    """Analyse a single case using the OpenAI API"""
    case_name = case_data.get('setting', 'Unknown Case').split('\n')[0]
    
    prompt = f"""Analyse the following murder case carefully and determine who is the most likely killer.
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
4. Behavioural clues  # British spelling

After analysing all factors, conclude with the most probable killer and a brief explanation.
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a world-class detective analysing complex murder cases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return f"Case {case_number} Analysis:\n{response.choices[0].message.content}\n"

def save_results_to_file(results, filename=None):
    """Save analysis results to a text file with timestamp"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"case_analysis_results_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("MURDER CASE ANALYSIS REPORT\n")
        file.write(f"Generated on: {datetime.now().strftime('%d %B %Y at %H:%M')}\n")
        file.write("="*50 + "\n\n")
        file.write(results)
    
    return filename

def main():
    cases = load_case_files()
    all_results = ""
    
    for i, case_data in enumerate(cases, 1):
        print(f"\n{'='*40}")
        print(f"Analysing Case {i}...")
        case_analysis = analyse_case(case_data, i)
        print(case_analysis)
        all_results += case_analysis + "\n" + "="*40 + "\n\n"
    
    # Save all results to file
    output_file = save_results_to_file(all_results)
    print(f"\nAnalysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main()