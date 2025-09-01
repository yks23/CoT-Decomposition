import os
import re
import sys
import json
import traceback
from datasets import load_dataset, Dataset, concatenate_datasets
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, Manager

# --- 1. 配置 ---
NUM_PROCESSES =  50
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not API_KEY:
    print("错误：请设置 SILICONFLOW_API_KEY 环境变量。")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1",
)

MODEL_TO_USE = "Qwen/Qwen3-32B"
HF_DATASET_NAME = "deepmath"
HF_DATASET_CONFIG = "default"
OUTPUT_FILE = "deepmath.parquet"
PROGRESS_FILE = "progress.jsonl"
NUM_SAMPLES_TO_PROCESS = None

with open('prompt_1.txt', 'r', encoding='utf-8') as f:
    prompt_1 = f.read()
with open('prompt_2.txt', 'r', encoding='utf-8') as f:
    prompt_2 = f.read()
# --- 2. 辅助函数 ---

def create_transformation_prompt(question_text, solution_text, difficulty):
    """为API调用构建高质量的Prompt"""
    return prompt_1.replace("{question_text}", question_text).replace("{solution_text}", solution_text).replace("{difficulty}", str(difficulty))
def create_generation_prompt(question_text, solution_text, difficulty):
    return prompt_2.replace("{question_text}", question_text).replace("{solution_text}", solution_text).replace("{difficulty}", str(difficulty))
def worker_process(job):
    """这个函数由每个独立的进程执行。"""
    question, solution_text, final_answer, difficulty = job
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
    api_prompt = create_generation_prompt(question, solution_text, difficulty)  # Pass difficulty
    try:
        
        # import requests
        # url = "https://api.siliconflow.cn/v1/chat/completions"

        # payload = {
        #     "model": MODEL_TO_USE,
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": api_prompt
        #         }
        #     ]
        # }
        # headers = {
        #     "Authorization": f"Bearer {API_KEY}",
        #     "Content-Type": "application/json"
        # }

        # response = requests.post(url, json=payload, headers=headers)

        # print(response.json())
        # transformed_content = response.json()['choices'][0]['message']['content'].strip()
        response = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[{"role": "user", "content": api_prompt}],
            temperature=0.7, max_tokens=50000
        )
        transformed_content = response.choices[0].message.content.strip()+response.choices[0].message.think_content.strip()
        return {
            "status": "success", "question": question,
            "transformed_content": transformed_content, 'answer': final_answer
        }
    except Exception as e:
        print(f"Error processing {question}: {e}")
        return {"status": "failure"}

# --- 3. 主执行逻辑 ---
if __name__ == "__main__":
    freeze_support()
    completed_solution = []
    processed = set()
    print("--- 开始数据转换流程 (可续跑模式, V4格式) ---")
    
    if os.path.exists(PROGRESS_FILE):
        print(f"发现进度文件 {PROGRESS_FILE}，正在读取已完成的任务...")
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed.add(data['question'] + data['solution'])
        print(f"已加载 {len(processed)} 个已完成的任务。")

    print(f"正在从Hugging Face加载数据集: {HF_DATASET_NAME} (配置: {HF_DATASET_CONFIG})")
    dss = [Dataset.from_parquet(f'../d/train-0000{i}-of-00010.parquet') for i in range(10)]
    
    # Concatenate datasets using the correct method
    dataset_to_process = concatenate_datasets(dss)
    print(dataset_to_process[0])
    
    processed_before = len(completed_solution)
    
    jobs = []
    
    for item in dataset_to_process:
        # print(item)
        question = item['question']
        solution_text_1 = item['r1_solution_1']
        solution_text_2 = item['r1_solution_2']
        solution_text_3 = item['r1_solution_3']
        final_answer = item['final_answer']
        
        if (question, solution_text_1) not in processed:
            jobs.append((question, solution_text_1, final_answer, item['difficulty']))
        if (question, solution_text_2) not in processed:
            jobs.append((question, solution_text_2, final_answer, item['difficulty']))
        if (question, solution_text_3) not in processed:
            jobs.append((question, solution_text_3, final_answer, item['difficulty']))
        
    all_data = []
    last_save=0
    if not jobs:
        print("所有任务都已完成，或没有新的有效任务。")
    else:
        print(f"准备了 {len(jobs)} 个新的有效任务，现在开始并行处理...")
        
        # Using Manager to handle shared data like `processed`
        with Manager() as manager, Pool(processes=NUM_PROCESSES) as pool, open(PROGRESS_FILE, 'a', encoding='utf-8') as f_progress:
            for result in tqdm(pool.imap_unordered(worker_process, jobs), total=len(jobs), desc="正在并行转换"):
                print(result['status'])
                if result and result['status'] == 'success':
                    transformed_content = result['transformed_content']
                    question = result['question']
                    solution = result['transformed_content']
                    answer = result['answer']
                    all_data.append({
                        "question": question,
                        "solution": solution,
                        "answer": answer
                    })
                    print(len(all_data)-last_save)
                    if len(all_data)-last_save >= 100:
                        print(f"已成功处理 {len(all_data)} 条数据，最新问题: {question[:2]}...")
                        f_progress.write(json.dumps(all_data[-(len(all_data)-last_save):], ensure_ascii=False) + '\n')
                        f_progress.flush()
                        last_save = len(all_data)
    
    print("\n--- 数据转换全部完成！ ---")
    dataset = Dataset.from_list(all_data)
    print(f"正在保存转换后的数据集到 {OUTPUT_FILE} ...")
    dataset.to_parquet(OUTPUT_FILE)
