# # main_collab.py

# from collab import DetectiveDialogue

# if __name__ == "__main__":
#     dialogue = DetectiveDialogue()
#     dialogue.run_dialogue()


# main_collab.py

from collab import DetectiveDialogue

if __name__ == "__main__":
    dialogue = DetectiveDialogue(
        max_turns=10,
        delay=1,
        run_id="001",
        data_dir="data",
        case_file="cases/case1.yaml",
    )
    dialogue.run_dialogue()
