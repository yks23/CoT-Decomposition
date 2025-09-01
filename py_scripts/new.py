import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import tqdm
import json
import os
import torch
import re
from transformers import LogitsProcessor
import hydra
import statistics
from datasets import load_dataset,Dataset
from omegaconf import DictConfig, OmegaConf
def get_gsm8k():
# 下载并加载 GSM8K 数据集
    dataset = load_dataset("gsm8k", "main")  # "main" 是标准版本
    return dataset
def get_modified_gsm8k():
    dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k_modified/valid.parquet')
    return dataset
def get_math():
    dataset = load_dataset("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/hendrycks_math", "algebra")
    return dataset
def split_answer(answer,type='gsm8k'):
    """
    输入示例 answer 字符串格式：
        "思维链内容 #### 最终答案内容"
    返回：
        cot: 思维链字符串
        final_answer: 最终答案字符串
    """
    if type == 'gsm8k':
        parts = answer.split("####")
        if len(parts) != 2:
            # 不符合预期格式，直接返回原answer和空答案或None
            return answer.strip(), None
        
        cot, final_answer = parts
        return cot.strip(), final_answer.strip()
    elif type == 'math':
        match = re.findall(r"\\boxed\{([^}]*)\}", answer)
        if match:
            return answer,match[-1].strip()  # 通常最后一个就是最终答案
        return answer, None
    elif type == 'modified_gsm8k':
        return answer,answer.split("<answer>")[-1].strip()
class TemperatureScheduler(LogitsProcessor):
    """
    动态调整生成温度：
    - 前 first_tokens 使用 first_temp
    - 后续使用 second_temp
    - 遇到 special_seg 后切换温度，使用 flag 避免重复解码
    """
    def __init__(self, first_temp, second_temp, tokenizer, split_id):
        self.first_temp = first_temp
        self.second_temp = second_temp
        self.tokenizer = tokenizer
        self.triggered_flags = None  # batch_size 大小的 flag，在生成前初始化
        self.split_id = split_id

    def init_flags(self, batch_size):
        # 初始化每个样本的 flag
        self.triggered_flags = [False] * batch_size

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 如果 trigger_flags 还没初始化
        if self.triggered_flags is None or len(self.triggered_flags) != batch_size:
            self.init_flags(batch_size)

        for i in range(batch_size):
            temp = None

            # 还没触发 special_seg
            if not self.triggered_flags[i]:
                # 超过指定 token 后才检查
                if self.split_id in input_ids[i]:
                    self.triggered_flags[i] = True

            # 根据 flag 和 first_tokens 决定温度
            if self.triggered_flags[i]:
                temp = self.second_temp
            else:
                temp = self.first_temp

            # 调整 logits
            if temp == 0:
                argmax_token = torch.argmax(scores[i])
                new_scores = torch.full_like(scores[i], -1e9)
                new_scores[argmax_token] = 1e9
                scores[i] = new_scores
            elif temp != 1.0:
                scores[i] = scores[i] / temp

        return scores

def generate_and_compute_entropy(model, tokenizer, input_text, max_new_tokens, tp1,tp2,split_id=None,device='cuda'):
    # 编码输入
    input_ids = tokenizer(input_text, return_tensors="pt",padding=True, padding_side='left').to(device)
    # 使用模型进行生成
    model.eval()

# 设置参数，生成多个序列
    with torch.no_grad():
        outputs = model.generate(
                **input_ids,
                max_length=input_ids.input_ids.shape[1] + max_new_tokens,
                temperature=1.0,  # 控制分布的平滑度
                do_sample=True,           # 使用采样
            )
        decoded = tokenizer.batch_decode(outputs[:, input_ids.input_ids.shape[1]:], skip_special_tokens=True)
        
    return decoded 

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
def check_format(text:str):
    if text.count('<stochastic>') == 1 and text.count('<deterministic>') == 1 and text.count('<answer>') == 1:
            return True
    return False
def get_model(cfg,device):
    model_path = cfg.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.type == 'hf':
        return model,tokenizer
    if cfg.type=='fsdp':
        checkpoints = merge_model(cfg.checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
        return model,tokenizer
    if cfg.type == 'lora':
        raise NotImplementedError
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
    special_tokens+=["<stochastic>","<deterministic>","<answer>"]
    # 获取保留的特殊 token 的 ID
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    
    # 解码之前，过滤掉不在保留列表中的特殊 token
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    
    # 使用过滤后的 token_ids 进行解码
    decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
    return decoded_text

class Calculator():
    def __init__(self,begin_id,end_id):
        self.begin_id = begin_id
        self.end_id = end_id
    def calculate(self,tokens,entropies):
        
        
        if self.begin_id==None:
            start = 0
        else:
            try:
                start = tokens.index(self.begin_id)
            except Exception as e:
                return 0,0,0
        if self.end_id==None:
            end = len(tokens) - 1
        else:
            try:
                end = tokens.index(self.end_id)
            except Exception as e:
                return 0,0,0
        
        length = end - start + 1
        entropies = sum(entropies[start:end+1])
        
        return length, entropies,entropies / length if length > 0 else 0


def calculate(tokens,entropy,start_id,end_id):
    lengths=[]
    sum_entropy = []
    avg_entropy = []
    print("start_id:", start_id)
    print("end_id:", end_id)
    for i in range(len(tokens)):
        leng,s,a = Calculator(start_id, end_id).calculate(tokens[i], entropy[i])
        if leng == 0:
            print(f"Sample {i+1} has no valid content.")
            continue
        lengths.append(leng)
        sum_entropy.append(s)
        avg_entropy.append(a)
        print(f"Sample {i+1} Length: {len}, Sum Entropy: {s}, Average Entropy: {a}")
    print("Average Length:", statistics.mean(lengths) if lengths else 0)
    print("Average Sum Entropy:", statistics.mean(sum_entropy) if sum_entropy else 0)
    print("Average Entropy per Token:", statistics.mean(avg_entropy) if avg_entropy else 0)
    return statistics.mean(lengths) if lengths else 0, statistics.mean(sum_entropy) if sum_entropy else 0,statistics.mean(avg_entropy) if avg_entropy else 0

def check_resume(path,seed):
    
    result_path = os.path.join(path,f'result_{seed}.json')
    static_path = os.path.join(path,f'static_{seed}.json')
    if os.path.exists(result_path) and os.path.exists(static_path):
        with open(result_path,'r') as f:
            results = json.load(f)
        with open(static_path,'r') as f:
            static = json.load(f)
        return results,static
    return None,None
@hydra.main(version_base=None, config_path="config", config_name="eval-llama")
def eval_main(cfg):
    rank = cfg.eval.rank
    # torch.cuda.set_device(rank)
    device = 'cuda'
    seed = cfg.eval.seed
    torch.manual_seed(seed)
    batchsize = cfg.eval.batch_size
    save_path = cfg.eval.save_path
    tp2 = cfg.eval.tp2
    tp1 = cfg.eval.tp1
    model,tokenizer = get_model(cfg.model,device)
    data_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/all_filtered.parquet'
    dataset = Dataset.from_parquet(data_path)
    results = []
    bar = tqdm.tqdm(total=len(dataset), desc="Processing questions",disable=(rank!=0))
    os.makedirs(save_path, exist_ok=True)
    minidatalength = len(dataset)//8
    for i in range(minidatalength*rank, minidatalength*(rank+1), batchsize):
            if i + batchsize > len(dataset):
                batchsize = len(dataset) - i
            bar.update(batchsize)
            questions = dataset[i:i + batchsize]['question']
            questions = [q.replace('\n<thinking>','')+'Please reasoning step by step' for q in questions]
            questions = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in questions]
            gt_answers = dataset[i:i + batchsize]['answer']
            gt_solutions = dataset[i:i + batchsize]['solution']
            all_results= generate_and_compute_entropy(model,tokenizer,questions,max_new_tokens=cfg.eval.max_new_tokens,tp1=tp1, tp2=tp2,split_id=None,device=device)
            for j in range(batchsize):
                results.append({
                    'question': questions[j],
                    'answer': gt_answers[j],
                    'solution': gt_solutions[j],
                    'reference': all_results[j],
                })
            if (i//batchsize) % 10 == 0:
                with open(os.path.join('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/',f'result_{rank}.json'), 'w') as f:
                    json.dump(results, f, indent=4)
if __name__ == "__main__":
    eval_main()