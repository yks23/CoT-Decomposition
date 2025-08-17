export CUDA_VISIBLE_DEVICES=3
python call_llm.py --model_name "llama3.1@8b" --output_path "./result/diverse-math-1.0-1.0" --temperature_strategy 0 --use_prompt --prompt_path ./preverification/zero-prompt2 --repeat 5 --tp1 1.0 --tp2 1.0 --dataset math
python call_llm.py --model_name "llama3.1@8b" --output_path "./result/diverse-math-0.0-0.0" --temperature_strategy 0 --use_prompt --prompt_path ./preverification/zero-prompt2 --repeat 1 --tp1 0.0 --tp2 0.0 --dataset math
