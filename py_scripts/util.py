import os
from typing import List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer

# 避免多进程并行死锁的警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def analyze_token_counts_parallel(texts: List[str], tokenizer_path: str) -> Tuple[float, int]:
    """
    使用多进程高效地计算字符串列表中所有文本的平均和总 token 数。

    Args:
        texts (List[str]): 待分析的字符串列表。
        tokenizer_path (str): 分词器（tokenizer）的路径或名称，例如 'bert-base-uncased'。

    Returns:
        一个元组，包含平均 token 数和总 token 数。
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("Input 'texts' must be a list of strings.")
    
    if not texts:
        return 0.0, 0

    # 将字符串列表转换为一个简单的 datasets.Dataset 对象
    dataset = Dataset.from_dict({'text': texts})
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from '{tokenizer_path}': {e}")
        return None, None

    # 定义一个函数，用于统计每个文本的 token 数量
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=False, padding=False)

    # 使用 map 方法进行多进程处理，并获取 token ID
    # num_proc 参数是实现多进程的关键
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count()//2,  # 使用所有可用的 CPU 核心
        remove_columns=['text']
    )

    # 统计每个文本的 token 数量
    total_tokens = sum(len(ids) for ids in tokenized_dataset['input_ids'])
    average_tokens = total_tokens / len(texts)
    
    return average_tokens, total_tokens
