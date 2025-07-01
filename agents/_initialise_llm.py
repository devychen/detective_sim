# use llama-3.1-405b-instruct

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.rate_limiters import InMemoryRateLimiter

# read system variables
import os
import dotenv

# choose any model, catalogue is available under https://build.nvidia.com/models
MODEL_NAME = "meta/llama-3.1-405b-instruct"

# this rate limiter will ensure we do not exceed the rate limit
# of 40 RPM given by NVIDIA
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2 / 60,  # 3 requests per minute (this model allows much less requests)
    check_every_n_seconds=5,  # wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=1  # controls the maximum burst size
)

llm = ChatNVIDIA(
    model=MODEL_NAME,
    api_key=os.getenv("NVIDIA_API_KEY"), 
    temperature=0,   # ensure reproducibility,
    rate_limiter=rate_limiter  # bind the rate limiter
)