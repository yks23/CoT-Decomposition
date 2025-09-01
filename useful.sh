tmux new -s a4
ssh g14
conda activate verl
cd "./Kaisen.Yang/CoT Decomposition/verl/verl/trainer"

export CUDA_VISIBLE_DEVICES=1
python call_llm.py model_name=llama3.1@8b output_path=./result/diverse-gsm8k-1.0-0.0-new number=0,1,2,3,4 tp1=1.0 tp2=0.0 --multirun





python eval.py --config-path config --config-name eval-llama-reg



#!/bin/bash/tick

# 初始化起始步骤
step=100
max_step=1000  # 假设最大步骤为1000，按需调整

# 无限循环，每小时执行一次任务
while true; do
    # 等待 1 小时
    sleep 4000
    # 生成对应的模型路径
    step=50
    model_path="/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/CoT/lora-openrl/global_step_${step}/huggingface"
    
    # 执行任务
    python3 eval.py seed=0 number=0 model_path=/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/CoT/lora-openrl/global_step_50/huggingface name="lora-50" dataset=[openrl] batch_size=50
    
    # 输出当前步骤已完成
    echo "Step ${step} completed, waiting for next hour..."
    
    # 步骤递增
    step=$((step + 100))

    # 如果步骤大于最大值，则重置
    if [ $step -gt $max_step ]; then
        step=100
    fi


done


1. 测lora的结果的效果

2. 开始RL
export CUDA_VISIBLE_DEVICES=2
python new_static.py model_path='/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct' save_path="./raw.json" starts="[PLAN,ROLL]" ends="[ROLL,ANSWER]"
export CUDA_VISIBLE_DEVICES=0
python new_static.py model_path="/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/CoT/lora-openrl/global_step_400/huggingface" save_path="./400.json" starts="[<stochastic>,<deterministic>]" ends="[<deterministic>,<answer>]"

python new_static.py --config-path=config --config-name=t2


ray start --head --port=6379
ray start --address='14.14.5.9:6379'


conda activate verl
cd "./Kaisen.Yang/CoT Decomposition/"


salloc -N 1 -n 1 -c 8 --gres=gpu:1 -p h01