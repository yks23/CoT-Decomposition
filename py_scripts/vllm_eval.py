import torch
import vllm
from vllm import SamplingParams
import torch.nn.functional as F
import os
import statistics
from transformers import AutoTokenizer
from datasets import Dataset
import hydra
import tqdm
import time
import json
# 计算给定 logits 的熵
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)  # 计算概率分布
    log_probs = torch.log(probs)  # 计算对数概率
    entropy = -torch.sum(probs * log_probs, dim=-1)  # 计算熵
    return entropy

# 生成文本并计算熵
def generate_and_compute_entropy(model, tokenizer, input_text, max_new_tokens, tp1, tp2, split_id=None):
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to('cuda')

    K = len(input_text)  # 生成序列的数量
    all_entropies = [[] for _ in range(K)]  # 每个序列的熵


    with torch.no_grad():
        # 使用 vllm 进行推理
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=tp1,
            top_p=tp2,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )

        # if split_id is not None:
        #     outputs = model.generate(
        #         input_ids,
        #         sampling_params=sampling_params,
        #         logit_processor=vllm.TemperatureScheduler(tp1, tp2, tokenizer, split_id)
        #     )
        # else:
        outputs = model.generate(
                input_ids,
                sampling_params=sampling_params
            )

        logits = outputs.scores  # 每一步生成的 logits
        tokens = outputs.sequences[:, input_ids.shape[1]:]  # 生成的 token 序列

        # 计算每个 token 的熵
        for step in range(len(logits)):
            for i in range(K):
                step_logits = logits[step][i]
                probs = torch.softmax(step_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                all_entropies[i].append(entropy.item())

    return tokens, all_entropies

# # 动态调整生成温度的 LogitsProcessor
# class TemperatureScheduler(vllm.logits_process.LogitsProcessor):
#     def __init__(self, first_temp, second_temp, tokenizer, split_id):
#         self.first_temp = first_temp
#         self.second_temp = second_temp
#         self.tokenizer = tokenizer
#         self.triggered_flags = None
#         self.split_id = split_id

#     def init_flags(self, batch_size):
#         self.triggered_flags = [False] * batch_size

#     def __call__(self, input_ids, scores):
#         batch_size = input_ids.shape[0]
#         if self.triggered_flags is None or len(self.triggered_flags) != batch_size:
#             self.init_flags(batch_size)

#         for i in range(batch_size):
#             temp = None
#             if not self.triggered_flags[i]:
#                 if self.split_id in input_ids[i]:
#                     self.triggered_flags[i] = True
#             temp = self.second_temp if self.triggered_flags[i] else self.first_temp
#             if temp != 1.0:
#                 scores[i] = scores[i] / temp

#         return scores

# 解码选定的特殊 token
def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    return tokenizer.decode(filtered_token_ids, skip_special_tokens=False)

# 加载并合并模型
def merge_model(model_path):
    if os.path.exists(os.path.join(model_path, 'full.safetensors')):
        return torch.load(os.path.join(model_path, 'full.safetensors'), weights_only=False)

    ckpts = {}
    world_size = 8
    shard_files = [os.path.join(model_path, f'model_world_size_8_rank_{i}.pt') for i in range(world_size)]

    for file_path in shard_files:
        tensors = torch.load(file_path, weights_only=False)
        for n, p in tensors.items():
            if n not in ckpts:
                ckpts[n] = p
            else:
                ckpts[n] = torch.cat([ckpts[n], p], dim=0)
    torch.save(ckpts, os.path.join(model_path, 'full.safetensors'))
    return ckpts

# 获取模型
def get_model(cfg):
    model_path = cfg.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = vllm.LLM(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# 计算长度、总熵、平均熵
class Calculator:
    def __init__(self, begin_id, end_id):
        self.begin_id = begin_id
        self.end_id = end_id

    def calculate(self, tokens, entropies):
        if self.begin_id is None:
            start = 0
        else:
            try:
                start = tokens.index(self.begin_id)
            except Exception as e:
                return 0, 0, 0
        if self.end_id is None:
            end = len(tokens) - 1
        else:
            try:
                end = tokens.index(self.end_id)
            except Exception as e:
                return 0, 0, 0

        length = end - start + 1
        entropies = sum(entropies[start:end + 1])

        return length, entropies, entropies / length if length > 0 else 0

# 计算平均长度、总熵和每个token的熵
def calculate(tokens, entropy, start_id, end_id):
    lengths = []
    sum_entropy = []
    avg_entropy = []

    for i in range(len(tokens)):
        leng, s, a = Calculator(start_id, end_id).calculate(tokens[i], entropy[i])
        if leng == 0:
            continue
        lengths.append(leng)
        sum_entropy.append(s)
        avg_entropy.append(a)

    return statistics.mean(lengths) if lengths else 0, statistics.mean(sum_entropy) if sum_entropy else 0, statistics.mean(avg_entropy) if avg_entropy else 0
def check_format(text:str):
    if text.count('<stochastic>') == 1 and text.count('<deterministic>') == 1 and text.count('<answer>') == 1:
            return True
    return False
@hydra.main(version_base=None, config_path="config", config_name="eval")
def eval_main(cfg):
    seed = cfg.eval.seed
    torch.manual_seed(seed)
    batchsize = cfg.eval.batch_size
    save_path = cfg.eval.save_path
    number = cfg.eval.number
    tp2 = cfg.eval.tp2
    tp1 = cfg.eval.tp1
    
    model,tokenizer = get_model(cfg.model)
    for dataset_name in cfg.eval.dataset:
        if dataset_name == 'gsm8k':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test.parquet'
        if dataset_name == 'math':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math-algebra/test.parquet'
        if dataset_name == 'aime24':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime24/default.parquet'
        if dataset_name == 'aime25':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime25/default.parquet'
        if dataset_name == 'amc23':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/amc23/default.parquet'
        if dataset_name == 'math500':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math500/default.parquet'
        if dataset_name == 'minerva':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/minerva/default.parquet'
        if dataset_name == 'olympiad_bench':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/olympiad_bench/default.parquet'
        if dataset_name =='openrl':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/test.parquet'
        if dataset_name == 'openrl-train':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/sub-train.parquet'
        if dataset_name == 'openrl-raw-test':
            data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/raw_test.parquet'
        if dataset_name.endswith('.parquet'):
            data_path = dataset_name
            dataset_name = dataset_name.split('/')[-1].split('.')[0]
        dataset = Dataset.from_parquet(data_path)
            
        time_start = time.time()
        results = []
        bar = tqdm.tqdm(total=len(dataset), desc="Processing questions")
        save_path = os.path.join(cfg.eval.save_path,f'{dataset_name}')
        os.makedirs(save_path, exist_ok=True)
        all_number = 0
        success_number = 0
        max_len= 1500
        format_success = 0
        
        lens=[]
        avgs=[]
        sums=[]
        
        for i in range(0, min(len(dataset),max_len), batchsize):
                if i + batchsize > len(dataset):
                    batchsize = len(dataset) - i
                bar.update(batchsize)
                questions = dataset[i:i + batchsize]['question']
                # questions = [q + 'please think step by step.give the answer at the lastline.' for q in questions]
                if dataset_name !='openral' and dataset_name != 'openrl-train':
                    questions = [q+cfg.eval.question_suffix  if cfg.eval.question_suffix != "" and not q.endswith(cfg.eval.question_suffix)  else q for q  in questions]
                    questions = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in questions]
                gt_answers = dataset[i:i + batchsize]['answer']
                    
                all_tokens,entropy = generate_and_compute_entropy(model,tokenizer,questions,max_new_tokens=cfg.eval.max_new_tokens,tp1=tp1, tp2=tp2,split_id=None)
                
                pred_result = [decode_with_selected_special_tokens(tokenizer,cfg.eval.head+cfg.eval.tail,ids) for ids in all_tokens]
                answers = [answer.split(cfg.eval.answer_seg)[-1] for answer in pred_result]
                    
                for gt_answer, pred_answer,q,full_text in zip(gt_answers, answers,questions,pred_result):
                    meta ={}
                    meta['question'] = q
                    meta['format_success'] = check_format(full_text)
                    meta['answer'] = pred_answer
                    meta['gt_answer'] = gt_answer
                    meta['success'] = gt_answer in meta['answer']
                    meta['full_text'] = full_text
                    if meta['success']:
                        success_number += 1
                    if meta['format_success']:
                        format_success += 1
                    all_number += 1
                    results.append(meta)
                ent={}
                for start,end in zip(cfg.eval.head,cfg.eval.tail):
                    if start=='begin' and end =='end':
                        l,s,v = calculate(all_tokens,entropy,None,tokenizer.eos_token_id)
                    else:
                        id1=tokenizer.convert_tokens_to_ids(start)
                        id2=tokenizer.convert_tokens_to_ids(end)
                        l,s,v=calculate(all_tokens,entropy,id1,id2)
                    lens.append(l)
                    avgs.append(v)
                    sums.append(s)
                    ent[start+end]={}
                    ent[start+end]['len'] = sum(lens)/len(lens)
                    ent[start+end]['sum'] = sum(sums)/len(sums)
                    ent[start+end]['avg'] = sum(avgs)/len(avgs)
                print("一批的预测结果已经完成。")
                print(f"当前批次成功率: {success_number / all_number:.2%} ({success_number}/{all_number})")
                with open(os.path.join(save_path,f'result_{number}.json'), "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                static =[result['success'] for result in results]
                static = {"vector":static,"success_rate":sum(static)/len(static),"success_number":sum(static),"all_number":len(static),"fommat_success":format_success,"format_rate":format_success/len(static) if len(static) > 0 else 0,"entropy":ent
                        }
                with open(os.path.join(save_path,f'static_{number}.json'), "w") as f:
                    json.dump(static, f, ensure_ascii=False, indent=4)
                    
        time_end = time.time()
        print(f"总耗时: {time_end - time_start:.2f}秒")
        print(f"平均每题耗时: {(time_end - time_start) / len(questions):.2f}秒")
    
if __name__ == "__main__":
    eval_main()