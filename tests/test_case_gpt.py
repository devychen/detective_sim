# test_case_gpt.py 

import os
import openai
from dotenv import load_dotenv
import yaml
from datetime import datetime

# Set OpenAI API key
load_dotenv('openai_key.env')
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_case_files():
    """Load and parse only case1.yaml from the cases directory"""
    file_path = os.path.join('cases', 'CN_case1.yaml')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            case_data = yaml.safe_load(file)
            if case_data is None:
                raise ValueError("YAML file is empty or invalid")
            return [case_data['case']]  # Ensure returning a list containing a single case
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        raise

def analyse_case(case_data, case_number):
    """Analyse a single case using the OpenAI API"""
    case_name = case_data.get('setting', 'Unknown case').split('\n')[0]
    
    prompt = f"""Please carefully analyze the following murder case and determine the most likely perpetrator.
        Consider all evidence, motives, opportunities, and forensic findings. Explain your reasoning step by step.
        All information is correct. Suspects may withhold information but definitely do not lie.

        Case: {case_name}
        Victim: {case_data['victim']['name']}
        Suspects: {', '.join([s['name'] for s in case_data['suspects']])}

        Key evidence:
        - Crime scene: {case_data['crime_scene'].get('body_state', 'No information')}
        - Forensic findings: {case_data.get('forensic_evidence', {}).get('cause_of_death', 'No information')}
        - Timeline: {'; '.join(case_data['timeline'])}

        After analyzing all factors, provide the most likely perpetrator and a detailed explanation.
        """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a world-class detective analyzing complex murder cases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return f"Case {case_number} analysis:\n{response.choices[0].message.content}\n"

def save_results_to_file(results, filename=None):
    """Save analysis results to a text file with timestamp"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Case_Analysis_Results_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Murder Case Analysis Report\n")
        file.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        file.write("="*50 + "\n\n")
        file.write(results)
    
    return filename

def main():
    try:
        cases = load_case_files()
        all_results = ""
        
        for i, case_data in enumerate(cases, 1):
            print(f"\n{'='*40}")
            print(f"Analyzing case {i}...")
            case_analysis = analyse_case(case_data, i)
            print(case_analysis)
            all_results += case_analysis + "\n" + "="*40 + "\n\n"
        
        # Save all results to file
        output_file = save_results_to_file(all_results)
        print(f"\nAnalysis completed. Results saved to: {output_file}")
    except Exception as e:
        print(f"Program error: {e}")

if __name__ == "__main__":
    main()