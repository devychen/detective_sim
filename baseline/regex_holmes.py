import re
import csv
import os

def extract_holmes_quotes(input_file, output_file, max_quotes=100, resume=False):
    # 检查是否需要从上次停止的位置继续
    start_position = 0
    existing_quotes = set()
    quote_count = 0
    
    if resume and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                existing_quotes.add(row[1])
                quote_count = int(row[0])
        
        # 找到文件中最后出现的位置
        with open(input_file, 'r', encoding='utf-8') as file:
            last_quote = list(existing_quotes)[-1]
            text_so_far = ""
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                text_so_far += chunk
                if last_quote in text_so_far:
                    start_position = file.tell() - len(chunk) + text_so_far.find(last_quote)
                    break
    
    # 读取文本文件
    with open(input_file, 'r', encoding='utf-8') as file:
        if start_position > 0:
            file.seek(start_position)
        text = file.read()
    
    # 初始化结果列表
    results = []
    
    # 定义正则表达式匹配引号内的内容和前后5个词
    pattern = r'((?:\w+\W+){0,5})(["''])(.*?)\2((?:\W+\w+){0,5})'
    
    # 查找所有匹配的引号内容
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        if len(results) >= max_quotes:
            break
            
        before_quote = match.group(1).strip()
        quote_content = match.group(3).strip()
        after_quote = match.group(4).strip()
        
        # 合并引号前后的文本和引号内容
        full_context = f"{before_quote} \"{quote_content}\" {after_quote}".strip()
        
        # 检查前后5个词中是否包含sherlock或holmes（不区分大小写）
        if (re.search(r'\bsherlock\b', full_context, re.IGNORECASE) or 
            re.search(r'\bholmes\b', full_context, re.IGNORECASE)):
            
            # 获取完整的句子
            sentences = re.split(r'(?<=[.!?])\s+', full_context)
            
            for sentence in sentences:
                if ('"' in sentence or "'" in sentence) and sentence not in existing_quotes:
                    results.append(sentence.strip())
                    existing_quotes.add(sentence)
                    quote_count += 1
                    if len(results) >= max_quotes:
                        break
    
    # 写入CSV文件
    mode = 'a' if resume and os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':
            writer.writerow(['ID', 'Sentence'])
        
        for sentence in results:
            quote_count += 1
            writer.writerow([quote_count, sentence])
    
    print(f"本次提取 {len(results)} 条句子，总计 {quote_count} 条句子")
    print(f"结果已保存到 {output_file}")
    print(f"下次运行可以添加 resume=True 参数继续提取")

# 使用示例
input_txt = '_novels/holmes_novel.txt'  # 替换为你的TXT文件路径
output_csv = 'holmes_quotes.csv'    # 输出的CSV文件路径

# 第一次运行（提取最多100句）
extract_holmes_quotes(input_txt, output_csv, max_quotes=100)

# 后续运行（继续提取，从上次停止的地方开始）
# extract_holmes_quotes(input_txt, output_csv, max_quotes=1000, resume=True)