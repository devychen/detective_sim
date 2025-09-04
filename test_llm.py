# test_llm.py
from agents.base_agent import BaseAgent
from llm_config import get_llama_llm  # 使用你现有的 llm_config.py

# 创建测试 agent
agent = BaseAgent(name="TestAgent", llm=get_llama_llm())

# 短 prompt 测试，降低 max_tokens
prompt = "Hello!"

try:
    resp = agent.run(prompt, max_tokens=50)  # 只请求 50 token
    print("LLM Response:", resp)
except Exception as e:
    print("Error occurred while calling LLM:")
    print(e)
