import os
import json
import hydra
import torch
import tqdm
import statistics
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from reward import normalize_final_answer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# ======================
# 分布式初始化
# ======================
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
import numpy as np

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


# ======================
# 数据层
# ======================

max_token_dataset = {
    "gsm8k": 1000,
    "math": 1500,
    "aime24": 3000,
    "aime25": 3000,
    "amc23": 2000,
    "math500": 1500,
    "minerva": 2000,
    "olympiad_bench": 3000,
    "openrl": 1000,
    "dapo": 3000,
}
max_batch_size ={
    "gsm8k": 20,
    "math": 10,
    "aime24": 5,
    "aime25": 5,
    "amc23": 10,
    "math500": 10,
    "minerva": 10,
    "olympiad_bench": 5,
    "openrl": 20,
    "dapo": 5,
}

from transformers import LogitsProcessor

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

def load_dataset_by_name(name: str):
    mapping = {
        "gsm8k": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test-new.parquet",
        "math": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math-algebra/test-new.parquet",
        "aime24": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime24/raw.parquet",
        "aime25": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime25/default.parquet",
        "amc23": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/amc23/default.parquet",
        "math500": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math500/default-new.parquet",
        "minerva": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/minerva/default.parquet",
        "olympiad_bench": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/olympiad_bench/default-new.parquet",
        "openrl": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/test.parquet",
        "openrl-train": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/sub-train.parquet",
        "openrl-raw-test": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/raw_test.parquet",
        "dapo": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/eval.parquet",
    }
    if name.endswith(".parquet"):
        return Dataset.from_parquet(name), name.split("/")[-1].split(".")[0]
    return Dataset.from_parquet(mapping[name]), name


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
        return results, static
    return [],{}


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
        end_id = self.get_token_id(end_token)
        
        # 找到开始位置
        if start_token == 'begin':
            start_idx = 0
        elif start_id in tokens:
            start_idx = tokens.index(start_id)
        else:
            return None  # 开始标记不存在
        
        # 找到结束位置
        if end_token == 'end':
            end_idx = len(tokens) - 1
        elif end_id in tokens:
            end_idx = tokens.index(end_id)
        else:
            return None  # 结束标记不存在
        
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
            "confidences": compute_confidence_metrics(confidences[start_idx:end_idx + 1])
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
        model.load_state_dict(checkpoints, strict=False)
    elif cfg.type == "lora":
        raise NotImplementedError
    
    return model, tokenizer

def generate_and_compute_entropy(
    model,
    tokenizer,
    input_text,
    max_new_tokens,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=0.7,
    thinkinig=False,
    dtype=torch.bfloat16,
):
    """支持多次采样（高效版，一次前向生成多样本）"""
    # 批处理输入
    print(input_text[0])
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        padding_side="left"
    ).to(device)
    B = len(input_text)  # batch size

    model.eval()
    model.to(device)
    logit_processors = []
    if not thinkinig:
        logit_processors.append(ForceNextTokenProcessor(
            trigger_token_id=tokenizer.convert_tokens_to_ids("<think>"),
            forced_token_id=tokenizer.convert_tokens_to_ids("</think>")
        ))

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=sample_num,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=[tokenizer.eos_token_id] if stop_token is None else [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(stop_token)[0],
            ],
            logits_processor=logit_processors
        )
 
        logits = outputs.scores  # list[step](B*sample_num, vocab)
        sequences = outputs.sequences[:, input_ids.input_ids.shape[1]:]  # only new tokens

        total_size = B * sample_num
        assert sequences.shape[0] == total_size

        # 初始化按 sample 排列
        all_results = []

        for s in range(sample_num):
            # 每个 batch 的第 s 个 sample
            batch_tokens = sequences[s::sample_num]  # (B, L)
            
            # 计算 entropy: steps x B -> B x L
            step_entropies = []
            step_confidences = []
            for step, logit in enumerate(logits):
                step_logit = logit[s::sample_num]  # (B, vocab)
                step_prob = torch.softmax(step_logit, dim=-1)
                step_entropy = -torch.sum(step_prob * torch.log(step_prob + 1e-8), dim=-1)
                
                step_confidence = torch.topk(step_prob, k=5, dim=-1).values
                step_confidence = torch.mean(torch.log(step_confidence + 1e-8), dim=-1)
                
                step_confidences.append(step_confidence.cpu())  # (B,)
                step_entropies.append(step_entropy.cpu())  # (B,)

            # 转置 step_entropies -> B x L
            batch_entropy = torch.stack(step_entropies, dim=1).tolist()  # (B, L)
            batch_confidence = torch.stack(step_confidences, dim=1).tolist()  # (B, L)
            # 每个 sample 对应 batch 内每个输入的 (tokens, entropy)
            tokens= []
            entropies=[]
            confidences =[]
            for i in range(B):
                tokens.append(batch_tokens[i].cpu().tolist())
                entropies.append(batch_entropy[i])
                confidences.append(batch_confidence[i])
                

            all_results.append((tokens, entropies, confidences))

    return all_results




def decode_predictions(tokenizer, tokens, special_tokens, answer_seg):
    pred_texts = [
        decode_with_selected_special_tokens(tokenizer, special_tokens, ids) for ids in tokens
    ]
    answers = [txt[-200:] for txt in pred_texts]
    return pred_texts, answers


# def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
#     decoded_text = ""
#     for token_id in token_ids:
#         token = tokenizer.decode([token_id])
#         if token in special_tokens:
#             decoded_text += token
#         elif token.strip():
#             decoded_text += token
#     return decoded_text
def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
            """
            解码 token 序列，仅保留指定的特殊 token，移除其他特殊 token。

            参数:
            - tokenizer: Tokenizer 实例，用于解码和转换 token。
            - special_tokens: 需要保留的特殊 token 的字符串列表。
            - token_ids: 待解码的 token 序列。

            返回:
            - 过滤后的解码文本字符串。
            """
            # 获取保留的特殊 token 的 ID
            special_tokens +=['<THINKING>','</THINKING>','<EXPLORATION>','</EXPLORATION>','<EXECUTION>','</EXECUTION>']
            keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
            
            # 解码之前，过滤掉不在保留列表中的特殊 token
            filtered_token_ids = [
                token_id for token_id in token_ids
                if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
            ]
            
            # 使用过滤后的 token_ids 进行解码
            decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
            return decoded_text

# ======================
# 评估层
# ======================
def check_format(text, checklist):
    return all(key in text for key in checklist)
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

def evaluate_batch(questions, gt_answers, samples, cfg, tokenizer):
    """
    samples: List[ (tokens, entropies) ]  # 每次采样的结果
    """
    K = len(questions)
    sample_num = len(samples)

    # 初始化熵计算器
    entropy_calculator = EntropyCalculator(tokenizer)
    
    # 收集每次采样的 decode 结果
    all_pred_texts = []
    all_answers = []
    for tokens, _,_ in samples:
        pred_texts, answers = decode_predictions(tokenizer, tokens, cfg.eval.head + cfg.eval.tail, cfg.eval.answer_seg)
        all_pred_texts.append(pred_texts)
        all_answers.append(answers)

    # 计算熵统计（使用第全部个样本）
    entropy_stats = {}
    if samples:
        all_tokens = []
        all_entropies=[]
        all_confidences = []
        for tokens, entropies,confidences in samples:
            all_tokens.extend(tokens)
            all_entropies.extend(entropies)
            all_confidences.extend(confidences)
        for start_token, end_token in zip(cfg.eval.head, cfg.eval.tail):
            stats = entropy_calculator.calculate_batch_entropy_stats(
                (all_tokens, all_entropies,all_confidences),start_token, end_token
            )
            key = f"{start_token}_{end_token}"
            entropy_stats[key] = stats

    # 逐题评估
    results = []
    success_avg, success_best, format_avg = 0, 0, 0

    for i in range(K):
        per_sample_success = []
        per_sample_format = []
        per_sample_outputs = []

        for s in range(sample_num):
            pred = all_answers[s][i]
            
            tokens = samples[s][0][i]
            entropies = samples[s][1][i]
            confidences = samples[s][2][i]
            
            full_text = all_pred_texts[s][i]
            succ = (normalize_final_answer(gt_answers[i]) in normalize_final_answer(pred))
            fmt = check_format(full_text, cfg.eval.checklist)
            per_sample_success.append(succ)
            per_sample_format.append(fmt)
            per_sample_outputs.append({
                "answer": normalize_final_answer(pred),
                "full_text": full_text,
                "success": succ,
                "format_success": fmt,
                "static": entropy_calculator.calculate_entropy_stats((tokens, entropies,confidences),cfg.eval.head[0], cfg.eval.tail[0])
            })

        # 平均成功率：样本成功率的均值
        success_avg += sum(per_sample_success) / sample_num
        format_avg += sum(per_sample_format) / sample_num
        # 最好成功率：只要有一次成功就算成功
        success_best += 1 if any(per_sample_success) else 0

        results.append({
            "question": questions[i],
            "gt_answer": gt_answers[i],
            "samples": per_sample_outputs,
            "avg_success": sum(per_sample_success) / sample_num,
            "best_success": 1 if any(per_sample_success) else 0,
            
        })

    batch_stats = {
        "avg_success": success_avg / K,
        "best_success": success_best / K,
        "format_avg": format_avg / K,
        "all_number": K,
        "entropy_stats": entropy_stats  # 添加熵统计
    }
    
    return results, batch_stats


# ======================
# 日志层
# ======================
def log_batch_stats(batch_idx, batch_stats, rank):
    if rank != 0:
        return
        
    avg_rate = batch_stats["avg_success"]
    best_rate = batch_stats["best_success"]
    fmt_rate = batch_stats["format_avg"]
    
    print(f"\n[Batch {batch_idx}] 🎲 平均成功率: {avg_rate:.2%} | 最好成功率: {best_rate:.2%} | 格式率: {fmt_rate:.2%}")
    
    # 打印熵统计信息
    if "entropy_stats" in batch_stats:
        print("📊 熵统计:")
        for key, stats in batch_stats["entropy_stats"].items():
            if stats["sample_count"] > 0:
                print(f"   {key}: 长度={stats['avg_length']:.1f}, "
                      f"总熵={stats['avg_total_entropy']:.2f}, "
                      f"平均熵={stats['avg_entropy_per_token']:.4f} "
                      f"({stats['sample_count']}样本)")


# ======================
# 分布式数据加载
# ======================
def get_distributed_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    
    # 创建数据加载器
    indices = list(sampler)
    return [dataset[i] for i in indices], len(indices)


# ======================
# 结果合并
# ======================
def merge_results_from_all_ranks(save_path, seed, world_size):
    if torch.distributed.get_rank() != 0:
        return
        
    all_results = []
    all_static = {"avg_success": 0, "best_success": 0, "format_avg": 0, "all_number": 0, "entropy_stats": {}}
    total_samples = 0
    
    for rank in range(world_size):
        result_file = os.path.join(save_path, f"result_{seed}_rank{rank}.json")
        static_file = os.path.join(save_path, f"static_{seed}_rank{rank}.json")
        
        if os.path.exists(result_file) and os.path.exists(static_file):
            with open(result_file, "r") as f:
                results = json.load(f)
            with open(static_file, "r") as f:
                static = json.load(f)
            
            all_results.extend(results)
            
            # 合并统计信息
            if static["all_number"] > 0:
                weight = static["all_number"]
                all_static["avg_success"] = (all_static["avg_success"] * total_samples + static["avg_success"] * weight) / (total_samples + weight)
                all_static["best_success"] = (all_static["best_success"] * total_samples + static["best_success"] * weight) / (total_samples + weight)
                all_static["format_avg"] = (all_static["format_avg"] * total_samples + static["format_avg"] * weight) / (total_samples + weight)
                total_samples += weight
                
                # 合并熵统计
                for key, stats in static.get("entropy_stats", {}).items():
                    if key not in all_static["entropy_stats"]:
                        all_static["entropy_stats"][key] = stats.copy()
                    else:
                        old_stats = all_static["entropy_stats"][key]
                        total_sample_count = old_stats["sample_count"] + stats["sample_count"]
                        
                        if total_sample_count > 0:
                            all_static["entropy_stats"][key] = {
                                "avg_length": (old_stats["avg_length"] * old_stats["sample_count"] + stats["avg_length"] * stats["sample_count"]) / total_sample_count,
                                "avg_total_entropy": (old_stats["avg_total_entropy"] * old_stats["sample_count"] + stats["avg_total_entropy"] * stats["sample_count"]) / total_sample_count,
                                "avg_entropy_per_token": (old_stats["avg_entropy_per_token"] * old_stats["sample_count"] + stats["avg_entropy_per_token"] * stats["sample_count"]) / total_sample_count,
                                "sample_count": total_sample_count
                            }
    
    # 保存合并后的结果
    if all_results:
        with open(os.path.join(save_path, f"result_{seed}_merged.json"), "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(save_path, f"static_{seed}_merged.json"), "w") as f:
            json.dump(all_static, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 合并完成: 总共 {len(all_results)} 个样本")
        print(f"📊 最终统计: 平均成功率={all_static['avg_success']:.2%}, "
              f"最好成功率={all_static['best_success']:.2%}, "
              f"格式率={all_static['format_avg']:.2%}")


# ======================
# 主控
# ======================
@hydra.main(version_base=None, config_path="config", config_name="eval")
def eval_main(cfg):
    # 初始化分布式
    ddp_setup()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = f"cuda:{rank}"
    torch.manual_seed(cfg.eval.seed + rank)  # 不同rank使用不同的随机种子

    model, tokenizer = load_model(cfg.model, device)
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank])
    
    overall_stats = {}
    
    if 'all' in cfg.eval.dataset:
        cfg.eval.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    
    for dataset_name in cfg.eval.dataset:
        dataset, dataset_name = load_dataset_by_name(dataset_name)
        if cfg.eval.batch_size == -1:
            batch_size = max_batch_size.get(dataset_name, 10)
        else:
            batch_size = cfg.eval.batch_size
        if cfg.eval.max_new_tokens == -1:
            max_tokens = max_token_dataset.get(dataset_name, 1000)
        else:
            max_tokens = cfg.eval.max_new_tokens    
        
        
        
        save_path = os.path.join(cfg.eval.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        if cfg.eval.get('resume', False):
            results, static = check_resume(save_path, cfg.eval.seed, rank)
        else:
            results, static = [], {}

        # 获取当前rank的数据子集
        subset, subset_size = get_distributed_dataloader(dataset, batch_size, rank, world_size)
        
        if rank == 0:
            bar = tqdm.tqdm(total=len(dataset), desc=f"Processing {dataset_name}")
        else:
            bar = None
        if len(results) > 0 and bar is not None: 
            bar.update(len(results))  # 更新已处理的数量
        # 初始化熵统计累计
        cumulative_entropy_stats = {}
        
        for i in range(len(results), subset_size, batch_size):
            batch_data = subset[i:i + batch_size]
            # guidelines = [d.get('correct_explorations', []) for d in batch_data]
            guidelines = [d.get('exploration', '') for d in batch_data]
            # 随机选择一条
            # import random
            # guidelines = [random.choice(g) if len(g) > 0 else "" for g in guidelines]
            
            questions = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q["question"] + cfg.eval.question_suffix.replace("<guideline>", g)}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=cfg.eval.get("enable_thinking", False)
                ) + cfg.eval.get("solution_prefix", "").replace("<guideline>", g)
                for q, g in zip(batch_data, guidelines)
                
            ]
            print(questions[0])
            gt_answers = [d["answer"] for d in batch_data]

            # === 多次采样 ===
            samples = generate_and_compute_entropy(
                model.module, tokenizer, questions,  # 使用model.module访问原始模型
                max_tokens,
                device=device,
                stop_token=cfg.eval.get("stop_token", None),
                sample_num=cfg.eval.sample_num,
                temperature=cfg.eval.temperature,
                top_p=cfg.eval.top_p,
            )

            batch_results, batch_stats = evaluate_batch(questions, gt_answers, samples, cfg, tokenizer)
            results.extend(batch_results)

            # 更新统计
            if not static:
                static = {"avg_success": 0, "best_success": 0, "format_avg": 0, "all_number": 0, "entropy_stats": {}}
            
            # 更新准确率统计
            static["avg_success"] = (static["avg_success"] * static["all_number"] + batch_stats["avg_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["best_success"] = (static["best_success"] * static["all_number"] + batch_stats["best_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["format_avg"] = (static["format_avg"] * static["all_number"] + batch_stats["format_avg"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["all_number"] += batch_stats["all_number"]
            
            # 更新熵统计（加权平均）
            if "entropy_stats" in batch_stats:
                for key, new_stats in batch_stats["entropy_stats"].items():
                    if key not in static["entropy_stats"]:
                        static["entropy_stats"][key] = new_stats.copy()
                    else:
                        old_stats = static["entropy_stats"][key]
                        total_samples = old_stats["sample_count"] + new_stats["sample_count"]
                        
                        if total_samples > 0:
                            static["entropy_stats"][key] = {
                                "avg_length": (old_stats["avg_length"] * old_stats["sample_count"] + new_stats["avg_length"] * new_stats["sample_count"]) / total_samples,
                                "avg_total_entropy": (old_stats["avg_total_entropy"] * old_stats["sample_count"] + new_stats["avg_total_entropy"] * new_stats["sample_count"]) / total_samples,
                                "avg_entropy_per_token": (old_stats["avg_entropy_per_token"] * old_stats["sample_count"] + new_stats["avg_entropy_per_token"] * new_stats["sample_count"]) / total_samples,
                                "sample_count": total_samples
                            }

            # 保存当前rank的结果
            with open(os.path.join(save_path, f"result_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            with open(os.path.join(save_path, f"static_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
                json.dump(static, f, ensure_ascii=False, indent=4)

            log_batch_stats(i // cfg.eval.batch_size, batch_stats, rank)
            if rank == 0:
                print("累计统计:")
                log_batch_stats(i // cfg.eval.batch_size, static, rank)
                bar.update(cfg.eval.batch_size * world_size)

        overall_stats[dataset_name] = static
        
        # 等待所有rank完成当前数据集
        torch.distributed.barrier()
        
        # 合并所有rank的结果
        merge_results_from_all_ranks(save_path, cfg.eval.seed, world_size)
        
        if bar is not None:
            bar.close()

    # 保存整体统计
    if rank == 0:
        with open(os.path.join(cfg.eval.save_path, f"overall_static_{cfg.eval.seed}.json"), "w") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=4)

    # 清理分布式环境
    destroy_process_group()
if __name__ == "__main__":
    eval_main()
    
#  torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 eval.py --config-path config --config-name eval-rl

