import os
from dotenv import load_dotenv
from case_manager import CaseManager

def main():
    # Load environment variables
    load_dotenv("nvidia_key.env")
    
    # Initialize case manager with a case file
    case_manager = CaseManager("config/cases/murder_at_manor.yaml")
    
    print("=== Starting Detective Simulation ===")
    
    # Stage 1: Onboarding
    print("\n=== Stage 1: Initial Reports ===")
    initial_reports = case_manager.run_onboarding()
    for detective, report in initial_reports.items():
        print(f"\n{detective}'s Initial Report:")
        print(report)
    
    # Stage 2: Discussion
    print("\n=== Stage 2: Discussion ===")
    discussion = case_manager.run_discussion()
    for i, msg in enumerate(discussion):
        print(f"\nTurn {i//3 + 1}: {msg['speaker']}")
        print(msg['content'])
    
    # Stage 3: Reflection
    print("\n=== Stage 3: Final Reports ===")
    final_reports = case_manager.run_reflection()
    for detective, report in final_reports.items():
        print(f"\n{detective}'s Final Report:")
        print(report)

if __name__ == "__main__":
    main()