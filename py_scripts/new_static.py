import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import statistics
import hydra
import datasets
# def generate_and_compute_entropy(model, tokenizer, input_text, K=5, max_new_tokens=50, temperature=1.0, device="cuda"):
#     # 编码输入
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

#     # 初始化结果记录
#     all_tokens = [[] for _ in range(K)]  # 每个序列的 token
#     all_entropies = [[] for _ in range(K)]  # 每个序列的熵

#     # 使用模型进行生成
#     model.eval()
#     model.to(device)
#     with torch.no_grad():
#         # 生成 K 条样本
#         generated_ids = input_ids.repeat(K, 1)  # [K, L] 复制 K 次初始输入
#         ended = torch.zeros(K, dtype=torch.bool, device=device)  # 记录哪些序列已经结束
        
#         for step in range(max_new_tokens):
#             print(f"Step: {step}")
#             outputs = model(generated_ids)
#             logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits
            
#             # 计算概率分布
#             probs = torch.softmax(logits / temperature, dim=-1)  # 使用 temperature 调节分布

#             # 计算每个 token 的熵
#             entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # 每个 token 的熵

#             # 选择下一个 token，采样（可以改成 `argmax` 来选择最大概率 token）
#             next_token = torch.multinomial(probs, num_samples=1)  # [K, 1]
#             ended = ended | (next_token.squeeze() == tokenizer.eos_token_id)  # 标记是否结束
#             print(f"Ended count: {ended.sum().item()}")

#             # 将生成的 token 拼接到已生成的序列中
#             generated_ids = torch.cat([generated_ids, next_token], dim=-1)

#             # 保存当前步骤的 token 和熵
#             for i in range(K):
#                 if not ended[i]:  # 只保存未结束的序列
#                     all_tokens[i].append(next_token[i].item())  # 保存当前 token
#                     all_entropies[i].append(entropy[i].item())  # 保存当前熵
#             if step % 30 == 0:
#                 print(f"Generated tokens so far: {tokenizer.decode(all_tokens[0])}")
#             # 如果所有序列都结束，跳出循环
#             if ended.all():
#                 break
#     # 转置结果：每一步的 token 和熵按顺序拼接
#     return all_tokens, all_entropies

def generate_and_compute_entropy(model, tokenizer, input_text, K=5, max_new_tokens=50, temperature=1.0, device="cuda"):
    # 编码输入
    input_ids = tokenizer([input_text]*K, return_tensors="pt").to(device)

    # 初始化结果记录
    all_tokens = [[] for _ in range(K)]  # 每个序列的 token
    all_entropies = [[] for _ in range(K)]  # 每个序列的熵

    # 使用模型进行生成
    model.eval()
    model.to(device)

# 设置参数，生成多个序列
    with torch.no_grad():
        # 使用 generate 方法
        outputs = model.generate(
            **input_ids, 
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,  # 控制分布的平滑度
            do_sample=True,           # 使用采样
            return_dict_in_generate=True,  # 返回 logits
            output_scores=True        # 输出 logits 和其他信息
        )

        # 获取 logits 和每个 token 的概率
        logits = outputs.scores  # 每一步生成的 logits
        
        # 计算每个 token 的熵
        for step in range(len(logits)):
            for i in range(K):
                # 当前步骤的 logits
                step_logits = logits[step][i]  # [vocab_size]
                probs = torch.softmax(step_logits / temperature, dim=-1)  # 获取概率分布

                # 计算熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # 每个 token 的熵

                # 获取当前 token 和熵
                next_token = torch.argmax(probs, dim=-1).item()  # 选择最大概率的 token
                all_tokens[i].append(next_token)  # 保存当前 token
                all_entropies[i].append(entropy.item())  # 保存当前熵值
    return all_tokens, all_entropies

def merge_model(model_path):
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

def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
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
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    
    # 解码之前，过滤掉不在保留列表中的特殊 token
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    
    # 使用过滤后的 token_ids 进行解码
    decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
    return decoded_text


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

@hydra.main(config_path="config", config_name="raw")
def main(cfg):
    from datasets import Dataset,load_dataset
    # dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw_train.parquet')
    
    model,tokenizer = get_model(
        cfg.model_path
    )
    for st in cfg.starts+cfg.ends:
        test_id = tokenizer.convert_tokens_to_ids(st)
        print(st, test_id)
    results={"name":cfg.model_path}
    results['decoded'] = []
    
    if cfg.get("load_checkpoint", False):
        
        print("Loading checkpoint from:", cfg.checkpoint_path)
        if os.path.exists(cfg.checkpoint_path):
            ckpts = torch.load(cfg.checkpoint_path, weights_only=False)
        else: 
            ckpts = merge_model(
                os.path.dirname(cfg.checkpoint_path)
            )
            torch.save(ckpts, "./full.safetensors")
        model.load_state_dict(ckpts, strict=False)
        model.to("cuda")
    if cfg.get("use_prompt", False):
        prompt = r"""
        For each problem:  
    "PLAN": briefly plan your solution strategy. Think about the steps you need to solve the problem.as short as you can.
    "ROLL": solve the problem step by step according to your plan.
    "ANSWER": provide a concise answer as a number.
        """
    else:
        prompt=""
    problem = "Find the solution to the problem: If x^2 +x + 1 = 2, what is the value of x?\n<thinking>"  # 输入文本
    
    input_text = prompt+problem
    
    messages = [
    {"role": "user", "content": input_text}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                                               )
    tokens_1,entropy_1 = generate_and_compute_entropy(
        model, tokenizer, input_text=input_text,
        K=25, max_new_tokens=300, temperature=1.0, device="cuda"
    )
    tokens_2,entropy_2 = generate_and_compute_entropy(
        model, tokenizer, input_text=input_text,
        K=25, max_new_tokens=300, temperature=1.0, device="cuda"
    )
    tokens = tokens_1 + tokens_2
    entropy = entropy_1 + entropy_2
    for i in range(len(tokens)):
        results['decoded'].append(decode_with_selected_special_tokens(
            tokenizer,
            cfg.starts + cfg.ends,
            tokens[i]
        ))
        print(results['decoded'][-1])
    
    for start,end in zip(cfg.starts,cfg.ends):
        if start=='begin' and end =='end':
            l,s,v = calculate(tokens,entropy,None,tokenizer.eos_token_id)
        else:
            id1=tokenizer.convert_tokens_to_ids(start)
            id2=tokenizer.convert_tokens_to_ids(end)
            l,s,v=calculate(tokens,entropy,id1,id2)
        results[start+end]={}
        results[start+end]['len'] = l
        results[start+end]['sum'] = s
        results[start+end]['avg'] = v
    import json
    if not os.path.exists("./entropy_statics"):
        os.makedirs("./entropy_statics")
    with open(os.path.join("/home/fit/alex/Kaisen.Yang/CoT Decomposition/entropy_statics",os.path.basename(cfg.save_path)), 'w') as f:
            json.dump(results, f, indent=4)
            print(f"Results saved to {cfg.save_path}")
    
if __name__=="__main__":
    main()
    
# for i, t in enumerate(texts):
#     print(f"[Sample {i}] {t}\n{'-'*60}")
# print("front_entropy(mean over samples):",
#       sum([x for x in stats['front_entropy_mean_per_sample'] if x==x]) /  # 过滤 NaN
#       max(1, sum([1 for x in stats['front_entropy_mean_per_sample'] if x==x])))
# print("back_entropy(mean over samples):",
#       sum([x for x in stats['back_entropy_mean_per_sample'] if x==x]) /
#       max(1, sum([1 for x in stats['back_entropy_mean_per_sample'] if x==x])))
