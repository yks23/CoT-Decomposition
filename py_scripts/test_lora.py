from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# 加载预训练模型和tokenizer
model_name = "/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"  # 可以选择其他预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,               # 低秩矩阵的秩（rank）
    lora_alpha=16,      # LoRA的缩放因子
    lora_dropout=0.0,   # LoRA的dropout率
    task_type="CAUSAL_LM",  # 设置任务类型
)


# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印模型结构，查看LoRA是否成功应用
print(model)
import torch
ckpt = torch.load('/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/CoT/lora-16/global_step_100/model_world_size_4_rank_0.pt',map_location='cuda')
print(ckpt.keys())