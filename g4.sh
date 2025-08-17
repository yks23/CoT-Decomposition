export CUDA_VISIBLE_DEVICES=3
export global_step=100
export split=train
export type=sft-full
cd verl/verl/trainer

# python process_checkpoint.py project_name=cot-decompose-$type global_step=$global_step
python eval.py number=$CUDA_VISIBLE_DEVICES model_path="/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/cot-decompose-$type/test/global_step_$global_step/huggingface" name=full-$global_step-$split dataset=gsm8k split=$split
