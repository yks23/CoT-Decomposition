import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset

# ===== 简单数据集 =====
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]



def run_inference(local_rank, rank, world_size, datasets,args):
    model_name = args.model
    output_file = args.output
    save_key = args.savekey
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] initialized on GPU {local_rank}")
    if 'execution' in args.savekey:
    # prompt_template = r"""You are a helpful and precise math solver.\nFor each problem, I will give your a question and a brief guideline. You should solve the question using the guideline.\nIn the output, just give a clear and correct step-by-step calculation to reach the final answer\nNow solve the following problem:\nQuestion: <question>\nguideline: <content>\n"""
    # prompt_template = r"""You are a helpful and precise math solver.\nFor each problem, I will give your a question and a detailed solution. You should output a breif guideline for solving the problem, based on the given solution.\nIn the output, You just need to give a correct guideline, don't contain detailed calculation , but contain critical idea for solving the question.Don't output other content that is not related to the solution.\nNow solve the following problem:\nQuestion: <question>\nSolution: <content>\n"""
        prompt_template = r"""Solve this Question: <question>\nthink step by step.Give your final answer in the boxed{} """
        max_new_tokens = 2500
        batch_size = 25
    elif 'exploration' in args.savekey:
        max_new_tokens = 500
        batch_size = 50
        prompt_template = r"""You are a precise math solver.
For each problem, I will give you a question and a detailed solution.
Your task is to write a **short and clear step-by-step guideline** for solving the problem.
Requirements:
1. Number the steps (format: 1. ,2., ...), making them easy to follow.
2. For each step, provide a concise plan of what to do.
3. Exclude **unrelated content**, just start with the steps.
4. Keep the guideline **short and concise**, without any **intermediate calculations and final answer**. You just need to provide a clear plan for solving, but not the detailed calculation.

Now solve the following problem:
Question: <question>
Solution: <content>
output the guideline below:
"""


    # 加载模型 & tokenizer
    
    # 数据划分 (DistributedSampler 会自动给每个 rank 一部分数据)
    dataset = PromptDataset(datasets)
    
    
    local_file = f"{output_file}_rank{rank}.json"
  
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    dataloader = DataLoader(dataset, sampler=sampler, num_workers=8,batch_size=batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        
    ).to(local_rank)
    model.eval()

    print(f"[Rank {rank}] starting inference on {len(sampler)} samples...")
    import tqdm
    if os.path.exists(local_file):
        results = json.load(open(local_file,'r'))
    else:
        results = []
    start_idx = len(results)//batch_size
    for i,batch in enumerate(tqdm.tqdm(dataloader, desc=f"Rank {rank} Inference", disable=rank!=0)):
        if i<start_idx:
            continue
        
        prompts = batch['question']
        try:
            contents = batch[args.contentkey]
        except:
            contents = ["" for _ in range(len(prompts))]
        prompts = [prompt_template.replace("<question>",q).replace("<content>",s) for q,s in zip(batch['question'], contents)]
        messages = [[{"role": "user", "content": text.replace("\n<thinking>","")}] for text in prompts]
        input_prompts = [tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False,enable_thinking=False) for msg in messages]
        inputs = tokenizer(input_prompts, return_tensors="pt", padding=True,padding_side='left').to(local_rank)
        prompt_length = inputs.input_ids.size(1)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=prompt_length + max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=[tokenizer.eos_token_id],
            )
            
        decoded = tokenizer.batch_decode(outputs[:,prompt_length:], skip_special_tokens=True)
        results +=[
            {**{k: v[i] for k, v in batch.items()}, save_key: new}
            for i, new in enumerate(decoded)
        ]

        print(f"[Rank {rank}] processed {len(results)} samples so far...")
        local_file = f"{output_file}_rank{rank}.json"
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    # 每个 rank 保存自己的结果
    local_file = f"{output_file}_rank{rank}.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    dist.barrier()
    if rank == 0:
        # rank=0 收集所有结果
        all_results = []
        for r in range(world_size):
            with open(f"{output_file}_rank{r}.json", "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ All results saved to {output_file}")
        dataset_new = HFDataset.from_list(all_results)
        dataset_new.to_parquet(output_file.replace('.json','.parquet'))
        for r in range(world_size):
            os.remove(f"{output_file}_rank{r}.json")
        
    dist.destroy_process_group()



if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct', help="HuggingFace 模型名")
    parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k", help="数据集名称")
    parser.add_argument("--output", type=str, default="./dataset/openrl/dataset_reference.json", help="输出文件")
    parser.add_argument("--savekey", type=str, default="", help="输出的键名，如果为空则保存所有键")
    parser.add_argument("--contentkey", type=str, default="exploration", help="输入的内容键名")
    args = parser.parse_args()
    
    args.model = "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    # ===== 从环境变量读取分布式信息 =====
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # dname = 'open-r1/OpenR1-Math-220k'
    # # summarize the dataset
    # dataset = load_dataset(dname,split="train")
    # d_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test.parquet'
    dataset = HFDataset.from_parquet(args.dataset)
    dataset = dataset.to_list()
    # args.output = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/new.json'
    # dataset =[{"problem": item['problem'], "answer":item['answer'], "solution":item['solution']} for item in dataset]
    # # dataset = dataset[:80]  # 仅测试用
    dist.init_process_group(backend="nccl", init_method="env://")
    run_inference(local_rank, rank, world_size, dataset, args)
    # dataset = HFDataset.from_parquet("./dataset/openrl/dataset_all.parquet")
    # dataset = dataset.to_list()
    # dist.init_process_group(backend="nccl", init_method="env://")
    # run_inference_2(local_rank, rank, world_size, args.model, dataset, args.output)
"""
torchrun --nproc_per_node=2 infer.py --dataset ./dataset/gsm8k/test.parquet --output ./dataset/gsm8k/test-new.json && torchrun --nproc_per_node=2 infer.py --dataset ./dataset/math-algebra/test.parquet --output ./dataset/math-algebra/test-new.json

torchrun --nproc_per_node=1 infer.py --dataset ./dataset/olympiad_bench/default.parquet --output ./dataset/olympiad_bench/default-new.json

torchrun --nproc_per_node=2 infer.py --dataset ./dataset/aime24/default.parquet --output ./dataset/aime24/default-new.json

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k.parquet --output ./dataset/s1k/s1k-new.json --savekey execution1

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k.parquet --output ./dataset/s1k/s1k-new.json --savekey execution2

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k.parquet --output ./dataset/s1k/s1k-new.json --savekey execution3

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey execution1 &&  torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey execution2 &&  torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey execution3 && torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey exploration1 &&  torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey exploration2 &&  torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey exploration3


torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey gemini_exploration --contentkey gemini_attempt && torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey deepseek_exploration --contentkey deepseek_attempt

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey execution1 --contentkey gemini_exploration && \
torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey execution2 --contentkey deepseek_exploration
    

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey exploration1 --contentkey execution1 && torchrun --nproc_per_node=8 infer.py --dataset ./dataset/s1k/s1k-new.parquet --output ./dataset/s1k/s1k-new.json --savekey exploration2 --contentkey execution2

torchrun --nproc_per_node=8 infer.py --dataset ./dataset/openrl/all_filtered.parquet --output ./dataset/openrl/qwen.json --savekey execution && torchrun --nproc_per_node=8 infer.py --dataset ./dataset/openrl/all_filtered.parquet --output ./dataset/openrl/qwen.json --savekey exploration --contentkey execution

"""