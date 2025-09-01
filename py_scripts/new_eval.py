import os
import json
import hydra
import torch
import tqdm
import statistics
from reward import normalize_final_answer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================
# 数据层
# ======================
def load_dataset_by_name(name: str):
    mapping = {
        "gsm8k": "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/correct.parquet",
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

def check_resume(path, seed):
    result_file = os.path.join(path, f"result_{seed}.json")
    static_file = os.path.join(path, f"static_{seed}.json")
    if os.path.exists(result_file) and os.path.exists(static_file):
        with open(result_file, "r") as f:
            results = json.load(f)
        with open(static_file, "r") as f:
            static = json.load(f)
        return results, static
    return None, None



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
    
    def calculate_entropy_stats(self, tokens, entropies, start_token, end_token):
        """
        计算特定标记区间的熵统计
        tokens: token id列表
        entropies: 对应的熵值列表
        start_token: 开始标记（字符串或'begin'/'end'）
        end_token: 结束标记（字符串或'begin'/'end'）
        """
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
        
        return {
            "length": length,
            "total_entropy": total_entropy,
            "avg_entropy": avg_entropy,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
    
    def calculate_batch_entropy_stats(self, all_tokens, all_entropies, start_token, end_token):
        """
        批量计算熵统计
        all_tokens: 所有样本的token列表
        all_entropies: 所有样本的熵值列表
        start_token, end_token: 区间标记
        """
        stats_list = []
        
        for tokens, entropies in zip(all_tokens, all_entropies):
            stats = self.calculate_entropy_stats(tokens, entropies, start_token, end_token)
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
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.type == "fsdp":
        checkpoints = merge_model(cfg.checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
    elif cfg.type == "lora":
        raise NotImplementedError
    
    return model, tokenizer


def generate_and_compute_entropy(model, tokenizer, input_text, max_new_tokens, device="cuda", stop_token=None, sample_num=1,temperature=1.0,top_p=0.7,enable_thinking=False):
    """支持多次采样"""
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, padding_side="left").to(device)
    K = len(input_text)

    model.eval()
    model.to(device)
    all_results = []

    for _ in range(sample_num):
        all_entropies = [[] for _ in range(K)]
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_length=input_ids.input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                enable_thinking=enable_thinking,
                output_scores=True,
                eos_token_id=[tokenizer.eos_token_id] if stop_token is None else [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids(stop_token),
                ],
            )
            logits = outputs.scores
            tokens = outputs.sequences[:, input_ids.input_ids.shape[1]:].tolist()

            for step in range(len(logits)):
                for i in range(K):
                    step_logits = logits[step][i]
                    probs = torch.softmax(step_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                    all_entropies[i].append(entropy.item())

            all_results.append((tokens, all_entropies))

    return all_results


def decode_predictions(tokenizer, tokens, special_tokens, answer_seg):
    pred_texts = [
        decode_with_selected_special_tokens(tokenizer, special_tokens, ids) for ids in tokens
    ]
    answers = [txt.split(answer_seg)[-1] for txt in pred_texts]
    return pred_texts, answers


def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
    decoded_text = ""
    for token_id in token_ids:
        token = tokenizer.decode([token_id])
        if token in special_tokens:
            decoded_text += token
        elif token.strip():
            decoded_text += token
    return decoded_text


# ======================
# 评估层
# ======================
def check_format(text, checklist):
    return all(key in text for key in checklist)


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
    for tokens, _ in samples:
        pred_texts, answers = decode_predictions(tokenizer, tokens, cfg.eval.head + cfg.eval.tail, cfg.eval.answer_seg)
        all_pred_texts.append(pred_texts)
        all_answers.append(answers)

    # 计算熵统计（使用第一个样本）
    entropy_stats = {}
    if samples:
        first_sample_tokens, first_sample_entropies = samples[0]
        for start_token, end_token in zip(cfg.eval.head, cfg.eval.tail):
            stats = entropy_calculator.calculate_batch_entropy_stats(
                first_sample_tokens, first_sample_entropies, start_token, end_token
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
            full_text = all_pred_texts[s][i]
            succ = (normalize_final_answer(gt_answers[i]) in normalize_final_answer(pred))
            fmt = check_format(full_text, cfg.eval.checklist)
            # is_pure_number = normalize_final_answer(pred).replace(",", "").replace(" ","")isdigit()
            per_sample_success.append(succ)
            per_sample_format.append(fmt)
            per_sample_outputs.append({
                "answer": pred,
                "full_text": full_text,
                "success": succ,
                "format_success": fmt,
                # "is_pure_number": is_pure_number
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
def log_batch_stats(batch_idx, batch_stats):
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
# 主控
# ======================
@hydra.main(version_base=None, config_path="config", config_name="eval")
def eval_main(cfg):
    device = "cuda"
    torch.manual_seed(cfg.eval.seed)

    model, tokenizer = load_model(cfg.model, device)
    overall_stats = {}
    
    if 'all' in cfg.eval.dataset:
        cfg.eval.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    
    for dataset_name in cfg.eval.dataset:
        dataset, dataset_name = load_dataset_by_name(dataset_name)
        save_path = os.path.join(cfg.eval.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        results, static = check_resume(save_path, cfg.eval.seed)
        if results is None:
            results, static = [], {}

        bar = tqdm.tqdm(total=len(dataset), desc=f"Processing {dataset_name}")
        
        # 初始化熵统计累计
        cumulative_entropy_stats = {}
        
        for i in range(0, len(dataset), cfg.eval.batch_size):
            batch = dataset[i:i + cfg.eval.batch_size]
            guidelines = batch.get('correct_explorations', [[]] * len(batch))
            # 随机选择一条
            import random
            guidelines = [random.choice(g) if len(g) > 0 else "" for g in guidelines]
            
            questions = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q + cfg.eval.question_suffix.replace("<guideline>", g)}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) + cfg.eval.get("solution_prefix", "").replace("<guideline>", g)
                for q, g in zip(batch["question"], guidelines)
            ]
            gt_answers = batch["answer"]

            # === 多次采样 ===
            samples = generate_and_compute_entropy(
                model, tokenizer, questions,
                cfg.eval.max_new_tokens,
                device=device,
                stop_token=None,
                sample_num=cfg.eval.sample_num,
                temperature=cfg.eval.temperature,
                top_p=cfg.eval.top_p,
                enable_thinking=cfg.eval.get("enable_thinking", False)
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

            # 保存
            with open(os.path.join(save_path, f"result_{cfg.eval.seed}.json"), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            with open(os.path.join(save_path, f"static_{cfg.eval.seed}.json"), "w") as f:
                json.dump(static, f, ensure_ascii=False, indent=4)

            log_batch_stats(i // cfg.eval.batch_size, batch_stats)
            print("累计统计:")
            log_batch_stats( i// cfg.eval.batch_size, static)
            bar.update(cfg.eval.batch_size)

        overall_stats[dataset_name] = static

    with open(os.path.join(cfg.eval.save_path, f"overall_static_{cfg.eval.seed}.json"), "w") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    eval_main()