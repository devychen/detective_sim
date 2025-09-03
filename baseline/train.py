import pandas as pd
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import torch

# =========================
# 1. 数据预处理
# =========================

def map_character(c):
    if c == "holmes":
        return "holmes"
    elif c == "poirot":
        return "poirot"
    elif c == "marple":
        return "marple"
    else:
        return "others"

def sample_by_tokens(df, tokenizer, max_tokens=5000):
    """
    从一个类别的数据中采样，直到累计 token 数接近 max_tokens。
    保证句子完整，不截断。
    """
    rows = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    total_tokens = 0
    sampled = []
    for _, row in rows.iterrows():
        quote = row["quote"]
        n_tokens = len(tokenizer.tokenize(quote))
        if total_tokens + n_tokens > max_tokens and len(sampled) > 0:
            break
        sampled.append(row)
        total_tokens += n_tokens
    return pd.DataFrame(sampled)

def prepare_datasets(csv_path, tokenizer, max_tokens=5000):
    df = pd.read_csv(csv_path)

    # 角色映射到4类
    df["label"] = df["character"].apply(map_character)

    # 数字化标签
    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # 按类别采样
    sampled_dfs = []
    for label in labels:
        part = df[df["label"] == label]
        sampled = sample_by_tokens(part, tokenizer, max_tokens=max_tokens)
        sampled_dfs.append(sampled)
    df_balanced = pd.concat(sampled_dfs).reset_index(drop=True)

    # 划分训练/验证集
    train_df, eval_df = train_test_split(
        df_balanced,
        test_size=0.1,
        stratify=df_balanced["label_id"],
        random_state=42
    )

    # 转换为 Huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset, label2id, id2label


# =========================
# 2. Tokenizer & 数据处理
# =========================

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# 3. 主训练逻辑
# =========================

def main():
    model_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 数据准备
    train_dataset, eval_dataset, label2id, id2label = prepare_datasets(
        "lines/train_lines.csv", tokenizer, max_tokens=5000
    )

    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_dataset = train_dataset.remove_columns(
        ["no.", "character", "quote", "label"]
    )
    eval_dataset = eval_dataset.remove_columns(
        ["no.", "character", "quote", "label"]
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./bert-classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 训练
    trainer.train()

    # 保存模型
    trainer.save_model("./bert-classifier")
    tokenizer.save_pretrained("./bert-classifier")

    # =========================
    # 4. 交互式测试
    # =========================
    classifier = pipeline("text-classification", model="./bert-classifier", tokenizer=tokenizer)

    print("\n✅ 模型训练完成，可以开始测试了。输入一句台词，模型会预测角色 (输入 exit 退出)\n")

    while True:
        text = input("台词: ")
        if text.strip().lower() == "exit":
            break
        pred = classifier(text, truncation=True, max_length=128)
        print("预测结果:", pred)


if __name__ == "__main__":
    main()
