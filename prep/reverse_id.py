import os
import dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
import yaml

# Load environment variables
dotenv.load_dotenv("nvidia_key.env")  # Ensure your API key is in this file

# Choose the model (check https://build.nvidia.com/models for available models)
MODEL_NAME = "meta/llama-3.3-70b-instruct"  # Updated model name

# System message for the detective analysis task
SYSTEM_MESSAGE = "You are an expert in literary analysis."

class DetectiveAnalyzer:
    def __init__(self):
        # Rate limiter to avoid hitting API limits (adjust as needed)
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=35 / 60,  # 35 requests per minute
            check_every_n_seconds=0.1,
            max_bucket_size=7,
        )
        self.llm = ChatNVIDIA(
            model=MODEL_NAME,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2,  # Slightly higher for varied but controlled responses
            rate_limiter=rate_limiter,
        )

    def analyze_detective(self, description):
        # Prepare the messages
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(
                content=f"""
                Identify which detective this description matches: 
                Sherlock Holmes, Hercule Poirot, or Miss Marple.
                Focus on linguistic patterns and personality traits.
                
                Description:
                {description}
                
                Respond with only the name.
                """
            ),
        ]
        # Get the AI's response
        response = self.llm.invoke(messages)
        return response.content.strip()

def main():
    # Load detective descriptions
    with open('prep_prompts/descriptions.yaml', 'r') as file:
        data = yaml.safe_load(file)
        descriptions = data['detectives']

    analyzer = DetectiveAnalyzer()
    print("Detective Identification Results:\n")
    
    for det_id, desc in descriptions.items():
        try:
            result = analyzer.analyze_detective(desc)
            print(f"{det_id}: {result}")
        except Exception as e:
            print(f"Error with {det_id}: {str(e)}")

if __name__ == "__main__":
    main()