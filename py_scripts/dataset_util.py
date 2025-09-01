import datasets
import os
import json
from datasets import Dataset
from tqdm import tqdm
# from matplotlib import pyplot as plt
import numpy as np
def load_dataset_by_name(name: str):
    mapping = {
        "gsm8k": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test-new.parquet",
        "math": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math-algebra/test-new.parquet",
        "aime24": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime24/default-new.parquet",
        "aime25": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime25/default.parquet",
        "amc23": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/amc23/default.parquet",
        "math500": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math500/default-new.parquet",
        "minerva": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/minerva/default.parquet",
        "olympiad_bench": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/olympiad_bench/default-new.parquet",
        "openrl": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/test.parquet",
        "openrl-train": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/sub-train.parquet",
        "openrl-raw-test": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/raw_test.parquet",
    }
    if name.endswith(".parquet"):
        return Dataset.from_parquet(name), name.split("/")[-1].split(".")[0]
    return Dataset.from_parquet(mapping[name]), name
def analyze_distribution(data,bins=20):
    """
    统计 0-1 区间数据分布，并可视化
    :param data: list[float]，数值在 [0,1] 内
    :param bins: 划分的区间数量
    :param show_plot: 是否画图
    """
    data = [i for i in data if filter(i)]
    data = [i['avg_success'] for i in data]
    data = np.array(data)
    max_ = np.max(data)
    for i in range(len(data)):
        data[i] = data[i]/max_
    # 检查范围
    if np.any(data < 0) or np.any(data > 1):
        raise ValueError("数据不在 [0,1] 区间内")
    hist, bin_edges = np.histogram(data, bins=bins, range=(0,1))
    for i in range(len(hist)):
        print(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {hist[i]}")
    plt.bar(bin_edges[:-1], hist, width=1/bins, edgecolor="black", align="edge")
    plt.xlabel("Value Range")
    plt.ylabel("Count")
    plt.title("Distribution in [0,1]")
    plt.savefig("./distribution.png")

def collect_all_correct_explorations(dataset_name):
    """
    批量处理所有 result_0.json, 构建 {idx -> [correct_explorations]} 的映射
    """
    # 找到所有相关文件
    possible_files = os.listdir("./evaluation/")
    all_jsons = []
    for name in possible_files:
        if not os.path.isdir(os.path.join("./evaluation", name)):
            continue
        all_datasets = os.listdir(os.path.join("./evaluation", name))
        if dataset_name in all_datasets:
            all_jsons.append(os.path.join("./evaluation", name, dataset_name, 'result_0.json'))

    # 累积统计
    idx_to_explorations = {}
    for json_file in all_jsons:
        with open(json_file) as f:
            data = json.load(f)
        print(f"Processing {json_file}, total {len(data)} samples.")
        for idx, item in enumerate(tqdm(data, desc=f"Reading {os.path.basename(json_file)}")):
            if 'samples' not in item:
                continue
            for sample in item['samples']:
                if sample['success']:
                    generation = item['question'] + sample['full_text']
                    if "<EXPLORATION>" not in generation or "</EXPLORATION>" not in generation:
                        continue
                    stochastic = generation.split("<EXPLORATION>")[-1].split("</EXPLORATION>")[0].strip()
                    if len(stochastic) == 0:
                        continue
                    idx_to_explorations.setdefault(idx, set()).add(stochastic)

    # 转换成 list
    idx_to_explorations = {k: list(v) for k, v in idx_to_explorations.items()}
    return idx_to_explorations


def make_correct_dataset_exploration(dataset_name, save_path=None):
    dataset, name = load_dataset_by_name(dataset_name)
    # 一次性解析所有文件
    idx_to_explorations = collect_all_correct_explorations(name)

    new_data = []
    possible_correct = 0
    for idx, item in enumerate(tqdm(dataset, desc=f"Building dataset {name}")):
        if idx not in idx_to_explorations:
            item['correct_explorations'] = []
            new_data.append(item)
            continue
        possible_correct += 1
        item['correct_explorations'] = idx_to_explorations[idx]
        
        new_data.append(item)

    if save_path is not None:
        new_dataset = Dataset.from_list(new_data)
        new_dataset.to_parquet(save_path)
        print(f"Saved to {save_path}, total {len(new_dataset)} samples.")
    
    cnt_num=[
        len(i['correct_explorations']) for i in new_data
    ]
    for data in new_data[:50]:
        print("Question:", data['question'])
        print("Answer:", data['answer'])
        print("Correct explorations:", len(data['correct_explorations']))
        print("-----")
    
    # analyze_distribution(cnt_num)
    print("Possible correct:", possible_correct, "/", len(dataset))
    


if __name__ == "__main__":
    make_correct_dataset_exploration("gsm8k","./dataset/gsm8k/correct.parquet")
    
    # make_correct_dataset_exploration("olympiad_bench","./dataset/olympiad_bench/correct-exploration.parquet")
