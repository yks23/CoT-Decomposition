import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig
from transformers import AutoModelForCausalLM

# 设置模拟的分布式环境
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
world_size = 8

dist.init_process_group(backend='nccl', rank=0, world_size=world_size)  # rank=0就行，单卡模拟

model_name = '/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct'

model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

# 构建模型
model.cuda()

# 包装成 FSDP（模拟多卡）
model = FSDP(model)

# 加载 8 个 shard 文件
shard_files = [f'/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/verl/trainer/checkpoints/verl_grpo_dapo/CoT/global_step_20/actor/model_world_size_8_rank_{i}.pt' for i in range(world_size)]
for i, f in enumerate(shard_files):
    # FSDP.load_state_dict() 只能在分布式 rank 对应 shard，所以单卡可以手动 load
    shard_state = torch.load(f, map_location='cuda:0')
    # 这里只是把每个 shard 的权重加载到对应参数
    # 单卡模拟不需要每个 rank 实际并行，只要能构建完整 DTensor 即可
    # 如果用 FSDP.full_state_dict()，它会自动 gather
    model.load_state_dict(shard_state, strict=False)

# 收集完整 tensor
full_state_dict = FSDP.full_state_dict(model, full_state_dict_config=FullStateDictConfig(offload_to_cpu=True))

# 保存单卡模型 checkpoint
torch.save(full_state_dict, 'full_model_single_gpu.pth')

dist.destroy_process_group()
