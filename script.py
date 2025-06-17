pip install python-dotenv
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API key safely loaded，length: {len(api_key)}")  # Use without printing out

# Codes for using the key...