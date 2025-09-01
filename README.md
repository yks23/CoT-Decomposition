# CoT-Decomposition

# Usage
## Environment Setup
Follow up the environment setup in verl

## SFT training
verl/verl/trainer/fsdp_sft_trainer.py

## RL training
verl/verl/trainer/main_ppo.py
the scripts is rl_train.sh and rl_qwen.sh

## dataset
we process the sft dataset by infer.py
openr1 is the dataset.
generally, we first sample response from base model(as execution part later), then extract the exploration part from the samples,and question+exploration+execution to train the sft model.

rl dataset is dapo-17k

evaluation dataset cotains:
gsm8k,MATH,math500,aime24,aime25,amc23,minerva,olympiad_bench