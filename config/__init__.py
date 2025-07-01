import os
from pathlib import Path

# 获取配置目录绝对路径
CONFIG_DIR = Path(__file__).parent

# 定义常用路径常量
PROMPTS_DIR = CONFIG_DIR / 'prompts'
CASES_DIR = CONFIG_DIR / 'cases'

__all__ = ['CONFIG_DIR', 'PROMPTS_DIR', 'CASES_DIR']

# 可选：添加配置验证逻辑
def _validate_configs():
    """检查必要的配置文件是否存在"""
    required = [
        PROMPTS_DIR / 'holmes_prompt.yaml',
        PROMPTS_DIR / 'poirot_prompt.yaml', 
        PROMPTS_DIR / 'marple_prompt.yaml'
    ]
    for f in required:
        if not f.exists():
            raise FileNotFoundError(f"Missing required config: {f}")

# 初始化时自动验证
try:
    _validate_configs()
except FileNotFoundError as e:
    print(f"Config Warning: {str(e)}")