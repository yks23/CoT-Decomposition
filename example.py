import os
import json
import hydra
import torch
import tqdm
import statistics
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from reward import normalize_final_answer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


# ======================
# 推理层
# ======================
def load_raw_model(model_path,model_type, device="cuda",checkpoint_path=None):
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="flash_attention_2",
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_type == "fsdp":
        
        checkpoints = torch.load(checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
    elif model_type == "lora":
        raise NotImplementedError
    
    return model, tokenizer


def do_test_sample(model,tokenizer,questions, max_new_tokens=2048, device="cuda"):
    inputs = tokenizer(questions, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens+input_ids.shape[1],
            do_sample=True,
            top_p=0.7,
            temperature=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    outputs = generation_output[:, input_ids.shape[1]:]
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs
if __name__ == "__main__":
    cfg_raw = {
        "model_path": "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B",# Path to your model
        "checkpoint_path": "",
        "type": "hf",  # huggingface 
    }
    cfg_sft = {
        "model_path": "/WORK/fit/alex/Kaisen/checkpoints/qwen/openr1-2/global_step_1000/huggingface",# Path to your model
        "checkpoint_path": "",
        "type": "hf",  # huggingface 
    }
    cfg_rl = {
        "model_path": "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B",# Path to your model
        "checkpoint_path": '/WORK/fit/alex/Kaisen/checkpoints/RL-Qwen/CoT/global_step_100/actor/full.safetensors',# Path to your checkpoint
        "type": "fsdp",  # fsdp
    }
    Test_Type = "raw" # raw, sft, rl
    if Test_Type == "raw":
        model, tokenizer = load_raw_model(**cfg_raw)
        test_question = ["Solve x^2 - 5x + 6 = 0. Provide the answer step by step."]
        
        # See think mode
        test_question_1 = tokenizer.apply_chat_template(test_question,enable_thinking=True)
        print(test_question_1)
        
        # See non-think mode
        test_question_2 = tokenizer.apply_chat_template(test_question,enable_thinking=False)
        print(test_question_2)
        
        # See normal generation
        answers = do_test_sample(model,tokenizer,test_question_1)
        answers = do_test_sample(model,tokenizer,test_question_2)
    if Test_Type == "sft":
        model, tokenizer = load_raw_model(**cfg_raw)
        test_question = ["Solve x^2 - 5x + 6 = 0. Provide the answer step by step."]
        
        # See think mode
        # test_question_1 = tokenizer.apply_chat_template(test_question,enable_thinking=True)
        # print(test_question_1)
        
        # See non-think mode
        test_question_2 = tokenizer.apply_chat_template(test_question,enable_thinking=False)
        test_question_2 = [q+"<EXPLORATION>" for q in test_question_2]
        
        # you need to add this special token <EXPLORATION> to your tokenizer
        
        print(test_question_2)
        
        # See normal generation
        # answers = do_test_sample(model,tokenizer,test_question_1)
        answers = do_test_sample(model,tokenizer,test_question_2)
    if Test_Type == "sft":
        model, tokenizer = load_raw_model(**cfg_raw)
        test_question = ["Solve x^2 - 5x + 6 = 0. Provide the answer step by step."]
        
        # See think mode
        # test_question_1 = tokenizer.apply_chat_template(test_question,enable_thinking=True)
        # print(test_question_1)
        
        # See non-think mode
        test_question_2 = tokenizer.apply_chat_template(test_question,enable_thinking=False)
        test_question_2 = [q+"<EXPLORATION>" for q in test_question_2]
        
        # you need to add this special token <EXPLORATION> to your tokenizer
        
        print(test_question_2)
        
        # See normal generation
        # answers = do_test_sample(model,tokenizer,test_question_1)
        answers = do_test_sample(model,tokenizer,test_question_2)
    
    print("Model and tokenizer loaded successfully.")