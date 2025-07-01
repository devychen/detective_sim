import yaml
from typing import Dict, List, Optional
from agents import SherlockHolmesAgent, HerculePoirotAgent, MissMarpleAgent

class CaseManager:
    def __init__(self, case_path: str):
        self.case = self._load_case(case_path)
        self.holmes = SherlockHolmesAgent()
        self.poirot = HerculePoirotAgent()
        self.marple = MissMarpleAgent()
        self.discussion_history = []
    
    def _load_case(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def distribute_initial_info(self):
        return {
            "Holmes": self.case["clues"]["holmes_clues"],
            "Poirot": self.case["clues"]["poirot_clues"],
            "Marple": self.case["clues"]["marple_clues"]
        }
    
    def run_onboarding(self) -> Dict[str, str]:
        initial_info = self.distribute_initial_info()
        return {
            "Holmes": self.holmes.generate_response(initial_info["Holmes"]),
            "Poirot": self.poirot.generate_response(initial_info["Poirot"]),
            "Marple": self.marple.generate_response(initial_info["Marple"])
        }
    
    def run_discussion(self, max_turns: int = 10) -> List[Dict]:
        initial_reports = self.run_onboarding()
        
        # Start discussion with initial reports
        self.discussion_history.append({"speaker": "Holmes", "content": initial_reports["Holmes"]})
        self.discussion_history.append({"speaker": "Poirot", "content": initial_reports["Poirot"]})
        self.discussion_history.append({"speaker": "Marple", "content": initial_reports["Marple"]})
        
        for turn in range(max_turns):
            # Rotate who speaks first each turn
            speakers = ["Holmes", "Poirot", "Marple"]
            if turn % 3 == 1:
                speakers = ["Poirot", "Marple", "Holmes"]
            elif turn % 3 == 2:
                speakers = ["Marple", "Holmes", "Poirot"]
            
            for speaker in speakers:
                agent = getattr(self, speaker.lower())
                response = agent.generate_response("", self.discussion_history)
                self.discussion_history.append({"speaker": speaker, "content": response})
                
                # Check if conclusion reached (simplified)
                if "I conclude" in response or "the murderer is" in response:
                    return self.discussion_history
            
        return self.discussion_history
    
    def run_reflection(self) -> Dict[str, str]:
        final_case_summary = "\n".join([f"{msg['speaker']}: {msg['content']}" 
                                      for msg in self.discussion_history])
        return {
            "Holmes": self.holmes.generate_response(final_case_summary),
            "Poirot": self.poirot.generate_response(final_case_summary),
            "Marple": self.marple.generate_response(final_case_summary)
        }