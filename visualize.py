import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set(style="whitegrid")

def smooth_curve(seq, chunk_size=5):
    """按 chunk_size 分块取均值"""
    n = len(seq)
    smoothed = []
    for i in range(0, n, chunk_size):
        smoothed.append(np.mean(seq[i:i+chunk_size]))
    return np.array(smoothed)

def load_results(result_path):
    """读取 result_*.json 文件"""
    if not os.path.isfile(result_path):
        dir_path = result_path
        result_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith("result") and f.endswith(".json")]
        results = []
        for file in result_files:
            with open(file, "r") as f:
                data = json.load(f)
                results.extend(data)
    else:
        with open(result_path, "r") as f:
            results = json.load(f)
    return results

def plot_entropy_confidence(all_entropies, all_confidences, output_dir):
    """绘制 单条样本+平均 的 entropy 和 confidence 曲线"""
    max_len = max(len(seq) for seq in all_entropies)

    # === Entropy ===
    entropy_matrix = np.zeros((len(all_entropies), max_len))
    mask_matrix = np.zeros((len(all_entropies), max_len))
    for i, seq in enumerate(all_entropies):
        seq = np.abs(seq)
        entropy_matrix[i, :len(seq)] = seq
        mask_matrix[i, :len(seq)] = 1
    avg_entropy = np.sum(entropy_matrix, axis=0) / np.sum(mask_matrix, axis=0)

    plt.figure(figsize=(12, 5))
    idx = np.random.choice(len(all_entropies))
    seq_entropy = np.abs(all_entropies[idx])
    plt.plot(seq_entropy, alpha=0.5, linewidth=1.5, label="Sample (raw)")
    plt.plot(smooth_curve(seq_entropy), alpha=0.7, linewidth=2, label="Sample (smoothed)")
    plt.plot(smooth_curve(avg_entropy), color='blue', linewidth=2.5, label="Average (smoothed)")
    plt.xlabel("Token Index")
    plt.ylabel("Entropy (abs)")
    plt.title("Entropy Across Token Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_over_idx.png"))
    plt.close()

    # === Confidence ===
    confidence_matrix = np.zeros((len(all_confidences), max_len))
    for i, seq in enumerate(all_confidences):
        seq = np.abs(seq)
        confidence_matrix[i, :len(seq)] = seq
    avg_confidence = np.sum(confidence_matrix, axis=0) / np.sum(mask_matrix, axis=0)

    plt.figure(figsize=(12, 5))
    idx = np.random.choice(len(all_confidences))
    seq_conf = np.abs(all_confidences[idx])
    plt.plot(seq_conf, alpha=0.5, linewidth=1.5, label="Sample (raw)")
    plt.plot(smooth_curve(seq_conf), alpha=0.7, linewidth=2, label="Sample (smoothed)")
    plt.plot(smooth_curve(avg_confidence), color='orange', linewidth=2.5, label="Average (smoothed)")
    plt.xlabel("Token Index")
    plt.ylabel("Confidence (abs)")
    plt.title("Confidence Across Token Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_over_idx.png"))
    plt.close()

def plot_success_distributions(step_nums, lengths, avg_entropies, avg_confidences, success_flags, output_dir):
    """绘制 success 与 stepnum/length/entropy/confidence 的分布图"""
    def hist_plot(data, success, fail, xlabel, title, filename):
        plt.figure(figsize=(10, 5))
        sns.histplot(success, color="green", label="Success", kde=False, stat="density", bins=20)
        sns.histplot(fail, color="red", label="Fail", kde=False, stat="density", bins=20)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    hist_plot(step_nums, step_nums[success_flags], step_nums[~success_flags],
              "Max Step Number", "Step Number Distribution by Success", "stepnum_success.png")
    hist_plot(lengths, lengths[success_flags], lengths[~success_flags],
              "Sequence Length", "Sequence Length Distribution by Success", "length_success.png")
    hist_plot(avg_entropies, avg_entropies[success_flags], avg_entropies[~success_flags],
              "Average Entropy per Sequence", "Average Entropy Distribution by Success", "avg_entropy_success.png")
    hist_plot(avg_confidences, avg_confidences[success_flags], avg_confidences[~success_flags],
              "Average Confidence per Sequence", "Average Confidence Distribution by Success", "avg_confidence_success.png")
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
def plot_by_task(results, output_dir, num_tasks=2, smooth_chunk=50):
    """按题目聚合，画出 num_tasks 个题目的平均轨迹"""
    task_entropies = []
    
    avg_entropy =[-item['avg_success']*torch.log2(torch.tensor(item['avg_success']+1e-10))-(1-item['avg_success'])*torch.log2(torch.tensor(1-item['avg_success']+1e-10)) for item in results]
    avg_entropy = sum(avg_entropy)/len(avg_entropy)
    
    print(f"所有题目的平均熵: {avg_entropy:.4f}")
    
    for item in results:  # 每个 item 就是一个题目
        seqs = []
       
        for sample in item["samples"]:
            static = sample["static"]
            if static is None:
                continue
            seqs.append(np.array(static["entropies"]))
        if seqs:
            task_entropies.append(seqs)

    # 选取前 num_tasks 个题目
    selected_tasks = task_entropies[:num_tasks]

    plt.figure(figsize=(12, 6))
    for tid, seqs in enumerate(selected_tasks, 1):
        max_len = max(len(seq) for seq in seqs)
        mat = np.zeros((len(seqs), max_len))
        mask = np.zeros((len(seqs), max_len))
        for i, seq in enumerate(seqs):
            mat[i, :len(seq)] = seq
            mask[i, :len(seq)] = 1
        avg_seq = np.sum(mat, axis=0) / np.sum(mask, axis=0)

        # 平滑处理：chunk 平均
        if smooth_chunk > 1:
            chunked = [
                np.mean(avg_seq[i:i+smooth_chunk])
                for i in range(0, len(avg_seq))
            ]
            avg_seq = np.array(chunked)

        plt.plot(avg_seq, linewidth=2, label=f"Task {tid}")

    plt.xlabel("Token Index")
    plt.ylabel("Entropy")
    plt.title(f"Average Entropy Trajectories of {num_tasks} Tasks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_entropy_trajectories.png"))
    plt.close()


def visualize_result_json(cfg):
    result_path = cfg.result_path
    output_dir = cfg.output_dir
    num_tasks = cfg.num_tasks
    results = load_results(result_path)

    all_entropies, all_confidences, step_nums, lengths, avg_confidences, success_flags = [], [], [], [], [], []
    for item in results:
        for sample in item["samples"]:
            static = sample["static"]
            if static is None:
                continue
            all_entropies.append(static["entropies"])
            all_confidences.append(static["confidences_all"])
            lengths.append(static["length"])
            avg_confidences.append(static["confidences"]["seq_avg_confidence"])
            step_nums.append(min(10, static["max_step"]))
            success_flags.append(sample["success"])

    step_nums = np.array(step_nums)
    lengths = np.array(lengths)
    avg_confidences = np.array(avg_confidences)
    avg_entropies = np.array([np.mean(seq) for seq in all_entropies])
    success_flags = np.array(success_flags)

    os.makedirs(output_dir, exist_ok=True)

    plot_entropy_confidence(all_entropies, all_confidences, output_dir)
    plot_success_distributions(step_nums, lengths, avg_entropies, avg_confidences, success_flags, output_dir)
    plot_by_task(results, output_dir, num_tasks=num_tasks)

    print(f"✅ 可视化完成，图片保存在: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True, help="Path to result")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--num_tasks", type=int, default=2, help="Number of tasks to plot")
    visualize_result_json(parser.parse_args())