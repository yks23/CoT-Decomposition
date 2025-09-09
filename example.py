import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
def merge_model(model_path):
    if os.path.exists(os.path.join(model_path, 'full.safetensors')):
        print("Merged model already exists.")
        return torch.load(os.path.join(model_path, 'full.safetensors'), weights_only=False)
    
    ckpts={}
    world_size = 8
    shard_files = [os.path.join(model_path,f'model_world_size_8_rank_{i}.pt') for i in range(world_size)]
    for file_path in shard_files:
        tensors = torch.load(file_path,weights_only=False)
        for n,p in tensors.items():
                if n not in ckpts:
                    p=p.to_local()
                    p = torch.tensor(p)
                    ckpts[n] = p
                else:
                    p=p.to_local()
                    p = torch.tensor(p)
                    ckpts[n] = torch.cat([ckpts[n],p],dim=0)
    torch.save(ckpts, os.path.join(model_path, 'full.safetensors'))
    return ckpts


# ======================
# 推理层
# ======================
def load_raw_model(model_path,type, device="cuda",checkpoint_path=None):
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="flash_attention_2",
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if type == "fsdp":
        checkpoints = merge_model(checkpoint_path)
        print("Loading checkpoint from:", checkpoint_path)
        model.load_state_dict(checkpoints, strict=False)
    elif type == "lora":
        raise NotImplementedError
    
    return model, tokenizer


def do_test_sample(model,tokenizer,questions, max_new_tokens=10000, device="cuda",stop_token=None):
    inputs = tokenizer(questions, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    if stop_token is not None:
        stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            
            max_new_tokens=max_new_tokens+input_ids.shape[1],
            do_sample=True,
            top_p=1.0,
            temperature=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[stop_token_id,tokenizer.eos_token_id] if stop_token is not None else tokenizer.eos_token_id,
        )
    outputs = generation_output[:, input_ids.shape[1]:]
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return decoded_outputs
if __name__ == "__main__":
    cfg_raw = {
        "model_path": "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B",# Path to your model
        "checkpoint_path": "",
        "type": "hf",  # huggingface 
    }
    cfg_sft = {
        "model_path": "/WORK/fit/alex/Kaisen/checkpoints/qwen/final/global_step_400/huggingface",# Path to your model
        "checkpoint_path": "",
        "type": "hf",  # huggingface 
    }
    """
    "/WORK/fit/alex/Kaisen/checkpoints/qwen/openr1-2/global_step_1000/huggingface"
    
    """
    cfg_rl = {
        "model_path": "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B",# Path to your model
        "checkpoint_path": '/WORK/fit/alex/Kaisen/checkpoints/RL-Qwen/Instruct/global_step_400/actor/',# Path to your checkpoint
        "type": "fsdp",  # fsdp
    }
    Test_Type = "rl" # raw, sft, rl
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
    if Test_Type == "rl":
        model, tokenizer = load_raw_model(**cfg_rl)
        # model.save_pretrained("./rl-400")
        # tokenizer.save_pretrained("./rl-400")
        test_question = "Problem: Compute S = sum from n=1 to infinity of 1 / [n²(n+1)].For these problem, you need to provide 2 different solutions methos, and combine them. notice that only one solution is not enough"
        # test_question = "Please answer the following multiple-choice question:\nAll of following are true about measles except -?\nA. Maximum incidence in 6m-3 year age group\nB. Best age for immunization is 9-12 months\nC. Secondary attack rate is 30%\nD. I.P. = 7-14 days. Provide the answer index step by step, give the answer in boxed"
        message = [
            {"role": "user", "content": test_question}
        ]
        
        print(tokenizer.convert_tokens_to_ids("execution"))
        
        print(message)
        input_str = tokenizer.apply_chat_template(message,enable_thinking=False,tokenize=False,add_generation_prompt=True)
        input_str += "<EXPLORATION>"
        # input_str += r"""""
        input_str = [input_str]
        # stop_token = "<EXECUTION>"
        stop_token = None
        answers = do_test_sample(model,tokenizer,input_str,stop_token=stop_token)
        print(answers)

    print("Model and tokenizer loaded successfully.")