tmux new -s a4
ssh g10
conda activate flow_matching
cd "Kaisen.Yang/CoT Decomposition"

export CUDA_VISIBLE_DEVICES=3
python call_llm.py model_name=llama3.1@8b output_path=./result/diverse-gsm8k-1.0-0.0-new number=0,1,2,3,4 tp1=1.0 tp2=0.0 --multirun
