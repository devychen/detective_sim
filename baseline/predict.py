# predict.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_dir = "./bert-classifier"

    # 加载保存好的模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    print("\n✅ 已加载训练好的模型，可以开始测试 (输入 exit 退出)\n")
    while True:
        text = input("台词: ")
        if text.strip().lower() == "exit":
            break
        pred = classifier(text, truncation=True, max_length=128)
        print("预测结果:", pred)

if __name__ == "__main__":
    main()
