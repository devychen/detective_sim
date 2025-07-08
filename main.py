# main.py

from agents.holmes_agent import HolmesAgent
from agents.poirot_agent import PoirotAgent
from agents.marple_agent import MarpleAgent

def main():
    # 初始线索（测试用）
    holmes_clue = """
你在现场观察到尸体仰面朝下倒在书房地板上，手中握着一张撕裂的纸条，上面写着：“他知道了。”
房间窗户紧闭，门上没有明显破坏痕迹。
"""

    poirot_clue = """
据目击者说，死者昨晚约在8点进入书房，之后没有人再见到他。
仆人听到书房中传来争执声，大约是在8:30到9:00之间。
"""

    marple_clue = """
死者是镇上的法官，与本地几位知名人物关系紧张。
其中一位嫌疑人——本杰明·克拉克——曾因该法官判决而失去财产，动机明显。
"""

    holmes = HolmesAgent()
    poirot = PoirotAgent()
    marple = MarpleAgent()

    print("🕵️ Sherlock Holmes 思考中...")
    print(holmes.run(holmes_clue), "\n")

    print("🕵️ Hercule Poirot 思考中...")
    print(poirot.run(poirot_clue), "\n")

    print("🕵️ Miss Marple 思考中...")
    print(marple.run(marple_clue), "\n")

if __name__ == "__main__":
    main()
