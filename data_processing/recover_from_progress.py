# recover_from_progress.py

import os
import json
from datasets import Dataset
from tqdm import tqdm

# --- 配置 ---
# 确保这些文件名与您之前运行的主脚本中的完全一致
# 注意: 请根据您最后一次运行的版本，确认是 v3 还是 v4
PROGRESS_FILE = "progress.jsonl"
OUTPUT_FILE = "OpenR1-Math-220k_sft_formatted_v4.parquet"

def main():
    print("--- 开始从进度文件恢复数据 ---")

    if not os.path.exists(PROGRESS_FILE):
        print(f"错误：进度文件 {PROGRESS_FILE} 不存在！无法进行恢复。")
        return

    print(f"正在从 {PROGRESS_FILE} 读取所有已处理的数据...")

    processed_results = []
    skipped_count = 0

    # 使用tqdm来显示读取进度
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        # 先获取总行数以正确显示进度条
        total_lines = sum(1 for line in f)
        f.seek(0) # 重置文件指针到开头

        for line in tqdm(f, total=total_lines, desc="正在读取进度"):
            try:
                processed_results.append(json.loads(line))
            except json.JSONDecodeError:
                skipped_count += 1

    if not processed_results:
        print("\n错误：未能从进度文件中读取到任何有效数据。")
        return

    print(f"\n成功读取 {len(processed_results)} 条有效数据。")
    if skipped_count > 0:
        print(f"发现并跳过了 {skipped_count} 个损坏的行。")

    print("\n正在创建新的Dataset对象...")
    final_dataset = Dataset.from_list(processed_results)

    print(f"正在将数据集保存到 Parquet 文件: {OUTPUT_FILE}")
    final_dataset.to_parquet(OUTPUT_FILE)

    print(f"\n恢复成功！最终文件已保存在: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
