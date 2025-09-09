
from transformers import LogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

import numpy as np
import statistics
def compute_confidence_metrics(confidences, chunk_size=100):
    """
    输入:
        confidences: list/ndarray, 每个token的confidence
        chunk_size: int, 每chunk的长度
    
    输出:
        dict, 包含四个指标
    """
    confidences = np.array(confidences)
    n = len(confidences)

    # 每个chunk的平均值
    chunk_means = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_means.append(confidences[start:end].mean())
    
    # 汇总指标
    results = {
        "seq_avg_confidence": confidences.mean(),
        "chunk_avg_confidences": chunk_means,
        "last_chunk_confidence": chunk_means[-1] if chunk_means else None,
        "min_chunk_confidence": min(chunk_means) if chunk_means else None,
    }
    return results
class ForceNextTokenProcessor(LogitsProcessor):
    def __init__(self, trigger_token_id, forced_token_id):
        self.trigger_token_id = trigger_token_id
        self.forced_token_id = forced_token_id
        self.active = False   # 是否触发

    def __call__(self, input_ids, scores):
        # input_ids: (batch, seq_len)
        last_token_id = input_ids[0, -1].item()

        # 如果上一个 token 是 trigger，就激活
        if last_token_id == self.trigger_token_id:
            self.active = True

        if self.active:
            # 把所有概率压到 forced_token_id 上
            mask = torch.full_like(scores, float("-inf"))
            mask[..., self.forced_token_id] = 0
            scores = mask
            self.active = False  # 只控制一步

        return scores

class FixedEntropyProcessorWithStop(LogitsProcessor):
    """
    通过温度控制输出分布熵为目标值，遇到分割token后停止熵控制
    """
    def __init__(self, tokenizer, target_entropy_1, target_entropy_2, stop_token, tol=0.01, max_iter=10):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            target_entropy_1 (float): 目标熵1
            target_entropy_2 (float): 目标熵2
            stop_token (str): 停止控制熵的token
            tol (float): 熵误差容忍度
            max_iter (int): 二分法最大迭代次数
        """
        self.tokenizer = tokenizer
        self.target_entropy_1 = target_entropy_1
        self.target_entropy_2 = target_entropy_2
        self.stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
        self.tol = tol
        self.max_iter = max_iter
        self.active = True  # 是否继续控制熵

    def __call__(self, input_ids, scores):
        # 检查最后一个 token 是否是 stop_token
        last_token_id = input_ids[0, -1].item()
        if last_token_id == self.stop_token_id:
            self.active = False
        
        if not self.active:
            self.target_entropy=self.target_entropy_2
        else:
            self.target_entropy=self.target_entropy_1
        if self.target_entropy is None:
            return scores

        logits = scores[0]  # 假设 batch=1
        # 二分法搜索温度
        low, high = 0.01, 10.0
        for _ in range(self.max_iter):
            T = (low + high) / 2
            probs = torch.softmax(logits / T, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
            if abs(entropy - self.target_entropy) < self.tol:
                break
            if entropy < self.target_entropy:
                high = T
            else:
                low = T
        
        # 调整 logits
        new_scores = logits / T
        return new_scores.unsqueeze(0)


def merge_model(model_path):
    if os.path.exists(os.path.join(model_path, 'full.safetensors')):
        print("Merged model already exists.")
        return torch.load(os.path.join(model_path, 'full.safetensors'), weights_only=False)
    
    ckpts={}
    world_size = 8
    shard_files = [os.path.join(model_path,f'model_world_size_8_rank_{i}.pt') for i in range(world_size)]
    for file_path in shard_files:
        tensors = torch.load(file_path,weights_only=False)
        for n,p in tensors.items():
                if n not in ckpts:
                    p=p.to_local()
                    p = torch.tensor(p)
                    ckpts[n] = p
                else:
                    p=p.to_local()
                    p = torch.tensor(p)
                    ckpts[n] = torch.cat([ckpts[n],p],dim=0)
    torch.save(ckpts, os.path.join(model_path, 'full.safetensors'))
    return ckpts

def check_resume(path, seed, rank):
    result_file = os.path.join(path, f"result_{seed}_rank{rank}.json")
    static_file = os.path.join(path, f"static_{seed}_rank{rank}.json")
    if os.path.exists(result_file) and os.path.exists(static_file):
        with open(result_file, "r") as f:
            results = json.load(f)
        with open(static_file, "r") as f:
            static = json.load(f)
        print(
            "[Rank {}] Resuming from existing results: {} samples found.".format(rank, len(results))
        )
        return results, static
    
    return [],{}
import re
def get_max_step(text: str) -> int:
    """
    提取文本中的序号步骤，返回最大序号（支持任意位数字）

    Args:
        text (str): 输入文本

    Returns:
        int: 文本中最大的序号，如果没有找到返回0
    """
    # 匹配任意位数字序号，格式如 1. 或 2)
    matches = re.findall(r'\b(\d+)[\.\)]', text)
    numbers = [int(num) for num in matches]
    return max(numbers) if numbers else 0

# ======================
# 熵统计工具类
# ======================
class EntropyCalculator:
    """用于计算特定标记区间的熵统计"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_token_id(self, token_str):
        """将标记字符串转换为token id"""
        if token_str == 'begin':
            return None  # 表示序列开始
        elif token_str == 'end':
            return None  # 表示序列结束
        else:
            return self.tokenizer.convert_tokens_to_ids(token_str)

    

    def calculate_entropy_stats(self, sample, start_token, end_token):
        """
        计算特定标记区间的熵统计
        tokens: token id列表
        entropies: 对应的熵值列表
        start_token: 开始标记（字符串或'begin'/'end'）
        end_token: 结束标记（字符串或'begin'/'end'）
        """
        tokens, entropies,confidences = sample
        
        start_id = self.get_token_id(start_token)
        # 找到开始位置
        if start_token == 'begin':
            start_idx = 0
            
        elif start_id in tokens:
            start_idx = tokens.index(start_id)
        else:
            return None  # 开始标记不存在
        
        if end_token == 'end':
            end_id = self.tokenizer.eos_token_id
        else:
            end_id = self.get_token_id(end_token)
            
        if end_id in tokens:
            end_idx = tokens.index(end_id)
        else:
            return None
        
        if self.tokenizer.eos_token_id in tokens:
            valid_end_pos = tokens.index(self.tokenizer.eos_token_id)
        else:
            valid_end_pos = len(tokens) - 1
        
        # 确保结束位置在开始位置之后
        if end_idx <= start_idx:
            return None
        
        # 提取区间内的熵值
        segment_entropies = entropies[start_idx:end_idx + 1]
        
        # 计算统计量
        length = len(segment_entropies)
        total_entropy = sum(segment_entropies)
        avg_entropy = total_entropy / length if length > 0 else 0
        
        max_step = get_max_step(self.tokenizer.decode(tokens[start_idx:end_idx + 1]))
        return {
            "length": length,
            "total_entropy": total_entropy,
            "avg_entropy": avg_entropy,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "max_step": max_step,
            "confidences": compute_confidence_metrics(confidences[start_idx:end_idx + 1]),
            "entropies": entropies[0:valid_end_pos + 1],
            "confidences_all": confidences[0:valid_end_pos + 1],
        }
    
    def calculate_batch_entropy_stats(self, sample,start_token, end_token):
        """
        批量计算熵统计
        all_tokens: 所有样本的token列表
        all_entropies: 所有样本的熵值列表
        start_token, end_token: 区间标记
        """
        all_tokens, all_entropies, all_confidences = sample
        
        stats_list = []
        
        for tokens, entropies,confidences in zip(all_tokens, all_entropies,all_confidences):
            stats = self.calculate_entropy_stats((tokens, entropies,confidences),start_token, end_token)
            if stats is not None:
                stats_list.append(stats)
        
        if not stats_list:
            return {
                "avg_length": 0,
                "avg_total_entropy": 0,
                "avg_entropy_per_token": 0,
                "sample_count": 0
            }
        
        # 计算平均值
        return {
            "avg_length": statistics.mean([s["length"] for s in stats_list]),
            "avg_total_entropy": statistics.mean([s["total_entropy"] for s in stats_list]),
            "avg_entropy_per_token": statistics.mean([s["avg_entropy"] for s in stats_list]),
            "sample_count": len(stats_list)
        }


# ======================
# 推理层
# ======================
def load_model(cfg, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="flash_attention_2",
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.type == "fsdp":
        checkpoints = merge_model(cfg.checkpoint_path)
        print("Loading checkpoint from:", cfg.checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
    elif cfg.type == "lora":
        raise NotImplementedError
    
    return model, tokenizer
