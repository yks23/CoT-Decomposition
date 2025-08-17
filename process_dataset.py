import pandas as pd
from datasets import Dataset
from preverification.dataset import get_math
from transformers import AutoTokenizer
def process_entry(entry):
    """
    处理单个数据条目，将其从原始格式转换为
    question: ... 和 answer: ... 的格式，并保留所有特殊标记。
    """
    try:
        question = entry['text'].split('\n<stochastic>')[0].strip()
        answer = entry['text'].split('<thinking>(\n')[1].strip()
        question=question.replace("\n\n<thinking>(","\n<thinking>")
        answer=answer.replace(')</thinking>','')
        answer=answer.replace('</stochastic>','')
        answer=answer.replace('</deterministic>','')
        answer=answer.replace('</answer>','')
        return {
                'question': question,
                'answer': answer
            }
    except Exception as e:
        print(f"处理失败的条目: {entry}. 错误: {e}")
        return None
    return None


def process_dataset_and_save(dataset_path, output_path):
    """
    读取数据集，处理每一行，然后保存为新的 .parquet 文件。
    """
    print(f"正在读取数据集: {dataset_path}")
    dataset = Dataset.from_parquet(dataset_path)

    print(f"正在处理数据集，总行数: {len(dataset)}")
    print(dataset[0])
    return
    # 使用 map 函数处理数据集
    processed_dataset = dataset.map(
        process_entry,
        remove_columns=['answer','question'],
        
    )
    
    # 移除处理失败的行
    processed_dataset = processed_dataset.filter(lambda x: x is not None)

    print(f"处理完成，新数据集总行数: {len(processed_dataset)}")
    
    # 保存为新的 .parquet 文件
    print(f"正在保存新的数据集到: {output_path}")
    processed_dataset.to_parquet(output_path)
    print(f"保存完成: {output_path}")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义输入和输出路径
    # train_input_path = './data_processing/OpenR1-Math-220k_sft_formatted_v4.parquet'
    
    train_output_path = './dataset/openrl/all.parquet'
    
    # dataset = []
    # import json
    # with open('/home/fit/alex/Kaisen.Yang/CoT Decomposition/data_processing/progress_v4.jsonl', 'r') as f:
    #     dataset = [json.loads(line) for line in f if line.strip()]
    # print(dataset[0])
    # final_dataset = []
    # for data in dataset:
    #     text = data['text']
    #     text= text.split('\n\nQ: ')[-1]
    #     question = text.split('\n\n<stochastic>')[0]+'\n<thinking>'
    #     answer='<stochastic>'+text.split('\n\n<stochastic>')[1]
    #     final_dataset.append({
    #         'question': question,
    #         'answer': answer
    #     })
    # print(f"处理后的数据集总行数: {len(final_dataset)}")
    # print(final_dataset[0]['question'])
    # print(final_dataset[0]['answer'])
    # parquet_dataset = Dataset.from_list(final_dataset)
    # parquet_dataset.to_parquet(train_output_path)
    
    
    test_load =  Dataset.from_parquet(train_output_path)
    
    
    # 随机 500 valid, 500 test, 剩下的 train
    test_output_path = './dataset/openrl/test.parquet'
    valid_output_path = './dataset/openrl/valid.parquet'
    train_output_path = './dataset/openrl/train.parquet'
    test_dataset = test_load.shuffle(seed=42).select(range(500))
    valid_dataset = test_load.shuffle(seed=42).select(range(500, 1000))
    train_dataset = test_load.shuffle(seed=42).select(range(1000, len(test_load)))
    test_dataset.to_parquet(test_output_path)
    valid_dataset.to_parquet(valid_output_path)
    train_dataset.to_parquet(train_output_path)
    # # 处理并保存训练集
    # process_dataset_and_save(train_input_path, train_output_path)
    
    
    # # --- 展示处理后的一个示例 ---
    # print("\n--- 处理后的一个示例 ---")
    # processed_train_dataset = Dataset.from_parquet(train_output_path)
    
    # # 1000条test，1000条valid，剩下的
    # processed_train_dataset = processed_train_dataset.shuffle(seed=42).select(range(3000))
    
    # test_output_path = './dataset/openrl/test.parquet'
    # valid_output_path = './dataset/openrl/valid.parquet'
    # train_output_path = './dataset/openrl/train.parquet'
    
    # processed_train_dataset.select(range(1000)).to_parquet(test_output_path)
    # processed_train_dataset.select(range(1000, 2000)).to_parquet(valid_output_path)
    # processed_train_dataset.select(range(2000, len(processed_train_dataset))).to_parquet(train_output_path)
    
    
    # # 取第一行数据
    # example = processed_train_dataset[0]
    
    # print("\n问题 (question):")
    # print(example['question'])
    # print("\n回答 (answer):")
    # print(example['answer'])
    
    
    # # print(dataset_math[0]['problem']+'\n'+'<thinking>')

    # # tokenizer_name = '/home/fit/alex/.cache/modelscope/hub/models/deepseek-ai/deepseek-math-7b-instruct'
    # # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # # encoded_question = tokenizer.encode(example['question'], add_special_tokens=False)
    # # encoded_answer = tokenizer.encode(example['answer'], add_special_tokens=False)
    # # print(encoded_answer)
    # # print(encoded_question)
    # # print("\n所有数据集处理和保存完成。")