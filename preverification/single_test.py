import re
import torch

def solve_single_question(question, solver):
    """
    给定一个单独的问题，用指定solver推理并返回模型输出和最终答案，方便调试。
    """
    messages = [
        {"role": "system", "content": solver.sys_prompt},
        {"role": "user", "content": question}
    ]
    text = solver.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = solver.tokenizer([text], return_tensors="pt", padding=True).to(solver.device)
    
    with torch.no_grad():
        outputs = solver.model.generate(**inputs, max_new_tokens=solver.max_new_tokens)
    
    full_text = solver.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    boxed_matches = re.findall(r"\\boxed\{(.+?)\}", full_text, re.DOTALL)
    final_answer = boxed_matches[-1].strip() if boxed_matches else None
    
    return full_text, final_answer
if __name__=="__main__":
    mode ='with'