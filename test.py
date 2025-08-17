import torch
from safetensors.torch import load_file, save_file
import os
import json

# -------------------
# 配置路径
# -------------------
safetensor_index = "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/cot-decompose-sft/test/global_step_1500/huggingface/model.safetensors.index.json"  # 索引文件
safetensor_dir = "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/cot-decompose-sft/test/global_step_1500/huggingface/"    # 分片所在目录
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# 读取索引
# -------------------
with open(safetensor_index, "r") as f:
    meta = json.load(f)  # 键名 -> 文件名
index = meta['weight_map']  # 获取权重映射

# -------------------
# 读取并合并每个 safetensor 文件
# -------------------
file_names = [file for file in os.listdir(safetensor_dir) if file.endswith('.safetensors')]

for file in file_names:
    file_path = os.path.join(safetensor_dir, file)
    
    # 加载索引文件中的键名
    if file not in index.values():
        continue  # 如果文件不在索引中，跳过
    
    tensors = load_file(file_path, device=device)  # 加载 safetensor 文件
    all_tensors = {}
    for key,value in tensors.items():
        new_key = key
        
        changed_key = key.replace("base_model.model.", "").replace(".base_layer.weight", ".weight")
        
        if ".base_layer.weight" in new_key:
            base_key = new_key
            # 找到对应的 LoRA A 和 B 权重
            lora_key_A = base_key.replace(".base_layer.weight", ".lora_A.default.weight")
            lora_key_B = base_key.replace(".base_layer.weight", ".lora_B.default.weight")
            
            if lora_key_A in tensors and lora_key_B in tensors:
                A = tensors[lora_key_A].detach()  # Detach to prevent gradient tracking
                B = tensors[lora_key_B].detach()

                # 合并 LoRA A 和 B 权重
                if A.shape[0] == B.shape[1]:
                    det = torch.matmul(B, A)
                else:
                    det = torch.matmul(A, B)

                base_tensor = tensors[base_key].detach()  # Detach to prevent gradient tracking
                shape_1 = base_tensor.shape
                if shape_1 == base_tensor.shape:
                    all_tensors[changed_key] = base_tensor + det
                else:
                    all_tensors[changed_key] = base_tensor + det.T
                    
                print(f"合并权重: {changed_key} (包含 LoRA 权重)")
            else:
                # 如果没有对应的 LoRA 权重，则仅保留 base 权重
                all_tensors[changed_key] = tensors[base_key].detach()  # Detach to free memory
                print(f"保留权重: {changed_key} (没有 LoRA 权重)")
        else:
            if 'lora' in new_key:
                print(tensors[key])
                continue
            else:
                print(f"保留权重: {changed_key} (非 LoRA 权重)")
                all_tensors[changed_key] = tensors[key].detach()  # Detach to free memory
    print(all_tensors.keys())
    del tensors
    print(f"合并文件: {file}，包含 {len(all_tensors)} 个权重")
    
    save_file(all_tensors, file_path)  # 保存合并后的权重

# -------------------
# 更新索引文件中的键名
# -------------------
new_index = {}
for key, file_name in index.items():
    # 修改键名 (如果需要的话)
    new_key = key
    if "base_model.model." in key:
        new_key = new_key.replace("base_model.model.", "")
    if ".base_layer.weight" in new_key:
        new_key = new_key.replace(".base_layer.weight", ".weight")
    if 'lora' in new_key:
        continue
    new_index[new_key] = file_name

# 更新索引文件
with open(safetensor_index, "w") as f:
    meta['weight_map'] = new_index
    json.dump(meta, f, indent=4)

print(f"Index file updated.")
