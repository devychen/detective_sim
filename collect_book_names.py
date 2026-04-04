import os
from pathlib import Path

# 根目录：_novels 所在目录（根据你实际路径调整）
ROOT_DIR = Path("_novels")

# 三个子文件夹
subfolders = ["holmes", "marple", "poirot"]

book_names = []

for folder in subfolders:
    folder_path = ROOT_DIR / folder
    if not folder_path.is_dir():
        continue
    for txt_file in folder_path.glob("*.txt"):
        # 去掉 .txt 扩展名
        name_without_ext = txt_file.stem
        # 将下划线替换为空格
        name_with_spaces = name_without_ext.replace("_", " ")
        # Title Case（书本大小写）
        title_cased = name_with_spaces.title()
        book_names.append(title_cased)

# 去重（如果需要）
book_names = list(dict.fromkeys(book_names))

# 写入到 book_names.txt
output_file = Path("book_names.txt")
with output_file.open("w", encoding="utf-8") as f:
    for name in book_names:
        f.write(name + "\n")

print(f"已写入 {len(book_names)} 个书名到 {output_file}")