import os
import re
import sys
import json
import traceback
from datasets import load_dataset, Dataset
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool, freeze_support

# --- 1. 配置 ---
NUM_PROCESSES = 64
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if not API_KEY:
    print("错误：请设置 SILICONFLOW_API_KEY 环境变量。")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1",
)

MODEL_TO_USE = "moonshotai/Kimi-K2-Instruct"
HF_DATASET_NAME = "open-r1/OpenR1-Math-220k"
HF_DATASET_CONFIG = "default"
OUTPUT_FILE = "OpenR1-Math-220k_sft_formatted_v4.parquet"
PROGRESS_FILE = "progress_v4.jsonl"
NUM_SAMPLES_TO_PROCESS = None

# --- 2. 辅助函数 ---

def create_transformation_prompt(question_text, solution_text):
    """为API调用构建高质量的Prompt"""
    return f"""
You are an expert in mathematical reasoning and data formatting. Your task is to transform a given math problem's solution into a new, clean format.
The format MUST be exactly three lines, each starting with a specific tag, with no extra symbols, parentheses, or closing tags:
<stochastic>A brief and intuitive idea for solving the problem
<deterministic>A clear and correct step-by-step explanation to reach the final answer
<answer>The final numerical or symbolic answer
Here is an example:
Original Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Original Solution: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.
Transformed Output:
<stochastic>Find May's sales (half of April's), then add them together for the total.
<deterministic>First, calculate the number of clips sold in May. Since it's half of the 48 clips sold in April, we compute 48 / 2 = 24 clips. Next, to find the total number of clips sold altogether, we add the sales from April and May: 48 + 24 = 72 clips.
<answer>72
---
Now, transform the following problem and solution into this exact three-line format.
Original Question:
{question_text}
Original Solution:
{solution_text}
Transformed Output:
"""

def worker_process(job):
    """这个函数由每个独立的进程执行。"""
    question, solution_text, final_answer = job
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
    api_prompt = create_transformation_prompt(question, solution_text)
    try:
        extra_params = { "enable_thinking": True }
        response = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[{"role": "user", "content": api_prompt}],
            temperature=0.1, max_tokens=2048, extra_body=extra_params
        )
        transformed_content = response.choices[0].message.content.strip()
        return {
            "status": "success", "question": question,
            "transformed_content": transformed_content, "final_answer": str(final_answer)
        }
    except Exception as e:
        return {"status": "failure"}

# --- 3. 主执行逻辑 ---
if __name__ == "__main__":
    freeze_support()
    print("--- 开始数据转换流程 (可续跑模式, V4格式) ---")
    
    completed_questions = set()
    if os.path.exists(PROGRESS_FILE):
        print(f"发现进度文件 {PROGRESS_FILE}，正在读取已完成的任务...")
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    match = re.search(r"^question: (.*?)\n<thinking>", data['text'], re.DOTALL)
                    if match:
                        completed_questions.add(match.group(1).strip())
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已加载 {len(completed_questions)} 个已完成的任务。")

    print(f"正在从Hugging Face加载数据集: {HF_DATASET_NAME} (配置: {HF_DATASET_CONFIG})")
    full_dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG)
    original_dataset = full_dataset['train']
    
    dataset_to_process = original_dataset
    if NUM_SAMPLES_TO_PROCESS is not None:
        dataset_to_process = original_dataset.select(range(NUM_SAMPLES_TO_PROCESS))

    jobs = []
    for entry in tqdm(dataset_to_process, desc="正在准备任务"):
        question = entry.get('problem')
        if question and question.strip() in completed_questions:
            continue
        solution_text = entry.get('solution')
        final_answer = entry.get('answer')
        if question and solution_text and final_answer is not None:
            jobs.append((question, solution_text, final_answer))

    if not jobs:
        print("所有任务都已完成，或没有新的有效任务。")
    else:
        print(f"准备了 {len(jobs)} 个新的有效任务，现在开始并行处理...")
        with Pool(processes=NUM_PROCESSES) as pool, open(PROGRESS_FILE, 'a', encoding='utf-8') as f_progress:
            for result in tqdm(pool.imap_unordered(worker_process, jobs), total=len(jobs), desc="正在并行转换"):
                if result and result['status'] == 'success':
                    final_answer = result['final_answer']
                    transformed_content = result['transformed_content']
                    question = result['question']
                    
                    # 修正: 使用 lambda 函数进行替换，彻底避免转义和引用问题
                    if re.search(r'^<answer>', transformed_content, re.MULTILINE):
                        transformed_content = re.sub(
                            r'^(<answer>).*$',
                            lambda m: m.group(1) + final_answer,
                            transformed_content,
                            flags=re.MULTILINE
                        )
                    else:
                        transformed_content += f"\n<answer>{final_answer}"
                    
                    sft_text_blob = (

                        "For each problem, you must output the solution in the following structured format:\n\n"
                        "<stochastic>A brief and intuitive idea for solving the problem\n"
                        "<deterministic>A clear and correct step-by-step explanation to reach the final answer\n"
                        "<answer>Final concise answer\n\n"
                        "Now solve the following problem:\n\n"
                        f"Q: {question}\n\n"
                        f"{transformed_content}"
                    )
                    f_progress.write(json.dumps({"text": sft_text_blob}) + '\n')
    
    print("\n--- 数据转换全部完成！ ---")

    print(f"正在从进度文件 {PROGRESS_FILE} 生成最终的 Parquet 文件...")
    processed_results = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    if not processed_results:
         print("\n错误：最终没有成功处理任何数据。")
         sys.exit(1)

    print(f"\n成功处理数据条目总数: {len(processed_results)}")
    
    print("\n正在创建新的Dataset对象...")
    final_dataset = Dataset.from_list(processed_results)
    
    print(f"正在将数据集保存到 Parquet 文件: {OUTPUT_FILE}")
    final_dataset.to_parquet(OUTPUT_FILE)

    print(f"\n成功！最终文件已保存在: {os.path.abspath(OUTPUT_FILE)}")
