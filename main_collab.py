# main_collab.py

from collab import DetectiveDialogue

if __name__ == "__main__":
    # 如需避免覆盖，把 append_timestamp=True
    dialogue = DetectiveDialogue(
        rule_path="rules/rule_collab.yaml",
        case_path="cases/case2.yaml",
        turns=10,
        log_file="data/dialogue_log.csv",
        prompt_file="data/prompts_log.txt",
        append_timestamp=False,   # 改 True 则文件名自动带时间戳
    )
    dialogue.run_dialogue()
