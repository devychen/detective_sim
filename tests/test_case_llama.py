import os
import requests
from dotenv import load_dotenv
import yaml
from datetime import datetime

# Set NVIDIA API key
load_dotenv('nvidia_key.env')
API_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"  # 修正模型名称(原代码中的3.3可能有误)
API_KEY = os.getenv("NVIDIA_API_KEY")

def load_case_files():
    """Load and parse only case1.yaml from the cases directory"""
    file_path = os.path.join('cases', 'CN_case1.yaml')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            case_data = yaml.safe_load(file)
            if case_data is None:
                raise ValueError("YAML file is empty or invalid")
            return [case_data['case']]  # 确保返回包含单个案例的列表
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        raise

def analyse_case(case_data, case_number):
    """Analyse a single case using the Llama API"""
    case_name = case_data.get('setting', '未知案件').split('\n')[0]
    
    prompt = f"""请仔细分析以下谋杀案并确定最可能的凶手。
        考虑所有证据、动机、机会和法医发现。逐步解释你的推理过程。
        信息全部都正确，嫌疑人提供的信息也许有隐瞒，但绝对没有撒谎。

        案件: {case_name}
        受害者: {case_data['victim']['name']}
        嫌疑人: {', '.join([s['name'] for s in case_data['suspects']])}

        关键证据:
        - 犯罪现场: {case_data['crime_scene'].get('body_state', '无信息')}
        - 法医发现: {case_data.get('forensic_evidence', {}).get('cause_of_death', '无信息')}
        - 时间线: {'; '.join(case_data['timeline'])}

        分析所有因素后，给出最可能的凶手和详细解释。
        """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一位世界级的侦探，正在分析复杂的谋杀案件。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    response_data = response.json()
    
    return f"案件 {case_number} 分析:\n{response_data['choices'][0]['message']['content']}\n"

def save_results_to_file(results, filename=None):
    """Save analysis results to a text file with timestamp"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"案件分析结果_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("谋杀案件分析报告\n")
        file.write(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n")
        file.write("="*50 + "\n\n")
        file.write(results)
    
    return filename

def main():
    try:
        cases = load_case_files()
        all_results = ""
        
        for i, case_data in enumerate(cases, 1):
            print(f"\n{'='*40}")
            print(f"正在分析案件 {i}...")
            case_analysis = analyse_case(case_data, i)
            print(case_analysis)
            all_results += case_analysis + "\n" + "="*40 + "\n\n"
        
        # Save all results to file
        output_file = save_results_to_file(all_results)
        print(f"\n分析完成。结果已保存至: {output_file}")
    except Exception as e:
        print(f"程序出错: {e}")

if __name__ == "__main__":
    main()