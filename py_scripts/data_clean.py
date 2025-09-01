import json
from vllm import LLM, SamplingParams
from datasets import load_dataset,Dataset,concatenate_datasets


# 1. 加载数据集（使用datasets库加载一个示例数据集）
# dataset_name = "zwhe99/DeepMath-103K"  # 请替换为你自己的数据集
# dataset = load_dataset(dataset_name)


dss = [Dataset.from_parquet(f'./d/train-0000{i}-of-00010.parquet') for i in range(10)]
    
    # Concatenate datasets using the correct method
dataset = concatenate_datasets(dss)

# 2. 根据prompt模板填充数据集条目
# 假设数据集有一个字段 'text'，将它和prompt模板组合起来
with open("prompt_2.txt", "r") as f:
    prompt = f.read()
prompts = [
    prompt.replace("<difficulty>",str(item['difficulty'])).replace('<question>',item['question']).replace('solution','rl_solution_1')  # 这里的 'text' 字段替换成你数据集中的实际字段
    for item in dataset  # 假设你从'train'集加载数据
]+[
    prompt.replace("<difficulty>",str(item['difficulty'])).replace('<question>',item['question']).replace('solution','rl_solution_2')  # 这里的 'text' 字段替换成你数据集中的实际字段
    for item in dataset  # 假设你从'train'集加载数据
]+[
    prompt.replace("<difficulty>",str(item['difficulty'])).replace('<question>',item['question']).replace('solution','rl_solution_3')  # 这里的 'text' 字段替换成你数据集中的实际字段
    for item in dataset  # 假设你从'train'集加载数据
]

# 3. 设置batch_size
batch_size = 32  # 你可以根据需要调整批次大小
batched_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]



# 4. 创建 LLM 实例
model_name = "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B"  # 假设模型名称为 Qwen3 32B

llm = LLM(model=model_name)
sampling_params = SamplingParams(max_tokens=5000,top_p=0.7)  # 使用默认设置
# 5. 生成输出
outputs = []
for batch in batched_prompts:
    batch_outputs = llm.generate(batch, sampling_params)
    for output in batch_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        outputs.append({
            "prompt": prompt,
            "generated_text": generated_text
        })
        print("generated_text:", generated_text)

# 6. 存储结果为 JSON
with open("generated_outputs.json", "w") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)

print("数据清洗并保存完毕！")
