import os
import json
import yaml
import requests
import dotenv
from pathlib import Path

# ========== CONFIG ==========
API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
TEMPLATE_PATH = "cases/case_template.yaml"
OUTPUT_PATH = "cases/test_case.yaml"
SCRIPTS_FOLDER = "_player_data/case1"  # 所有嫌疑人 JSON 文件的文件夹
dotenv.load_dotenv("nvidia_key.env")
API_KEY = os.getenv("NVIDIA_API_KEY")
# ============================


def load_suspect_scripts(folder_path):
    """读取所有嫌疑人的 JSON 文件，并提取脚本内容。"""
    scripts = []
    for file_path in Path(folder_path).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            script_text = "\n".join(data.get("script", []))
            scripts.append(f"# 嫌疑人：{file_path.stem}\n{script_text}")
    return "\n\n".join(scripts)


def load_case_template():
    """读取 YAML 模板作为输出格式指导。"""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def call_llm_api(prompt):
    """调用 NVIDIA LLM API。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一位严谨而聪明的一流侦探小说家，你正打算把剧本杀剧本作为素材，创作一部小说，你需要整理所有案件信息，你非常擅长梳理。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_case_yaml():
    """主逻辑流程。"""
    print("▶ 读取角色剧本...")
    all_scripts = load_suspect_scripts(SCRIPTS_FOLDER)

    print("▶ 读取案件模板...")
    template_text = load_case_template()

    prompt = f"""
    以下是几个剧本杀角色的背景资料，全部时间线已发生完毕，且所有线索均已揭晓，请你整理出一份面向侦探的完整案件描述。
    最终请严格按照 case_template.yaml 的结构输出结果。此外：
    1. 描述越详细越好，每一个空都需要填。
    2. 剧本只包含最关键信息，你需要根据这些事实，适当扩展。
    3. 时间线需要穷尽。
    4. appearance指的不仅仅是看起来是什么样的人，还包含身上的可疑证据，比如衣服痕迹之类的。
    5. key_items指的是搜身或者搜所在房间后的、和本案件密切相关的证物/观察

    # 模板：
    {template_text}

    # 剧本资料：
    {all_scripts}
    """

    print("▶ 正在调用 LLM 模型分析剧本...")
    llm_output = call_llm_api(prompt)

    print("▶ 正在保存结果到 case_info.yaml")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(llm_output.strip())

    print("✅ 案件分析完成，输出保存在 case_info.yaml")


if __name__ == "__main__":
    generate_case_yaml()
