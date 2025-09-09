import torch
from vllm import LLM, SamplingParams
from typing import List, Tuple

def calculate_entropy(token_logprob_dict):
    entropy = 0.0
    for id,logprob in token_logprob_dict.items():
        prob = torch.exp(torch.tensor(logprob.logprob))
        entropy -= prob * logprob.logprob
    return entropy.item()

def calculate_confidence(token_logprob_dict,K=5):
    sorted_logprobs = sorted([logprob.logprob for id, logprob in token_logprob_dict.items()], reverse=True)
    top_k_logprobs = sorted_logprobs[:K]
    confidence = -sum(top_k_logprobs)
    return confidence
def generate_and_get_logprobs(
    model: LLM,
    input_text: List[str],
    max_new_tokens: int,
    sample_num: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.7,
    stop_token: str = None
) -> List[Tuple[List[str], List[List[float]]]]:
    """
    使用 vllm 生成文本并获取每个生成 token 的对数概率。

    Args:
        model (LLM): vLLM 实例。
        input_text (List[str]): 输入提示的列表。
        max_new_tokens (int): 生成的新 token 的最大数量。
        sample_num (int): 每个提示生成的样本数量。
        temperature (float): 采样温度。
        top_p (float): top-p 采样值。
        stop_token (str): 停止生成的 token。

    Returns:
        一个元组列表。每个元组包含一个生成的文本列表和每个文本对应的 token 对数概率列表。
    """
    
    # 配置采样参数
    sampling_params = SamplingParams(
        n=sample_num,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=[model.get_tokenizer().convert_tokens_to_ids(stop_token)] if stop_token else None,
        logprobs=1  # 请求每一步生成的 token 的对数概率
    )

    # 生成文本和对数概率
    outputs = model.generate(input_text, sampling_params)

    results = []
    
    for output in outputs:
        generated_texts = []
        token_logprobs = []
        confidence_list = []
        # 获取批次内每个样本的结果
        for sequence in output.outputs:
            generated_texts.append(sequence.text)
            
            # logprobs 对象是字典的列表，每个字典对应一个 token
            logprob_list = [calculate_entropy(token) for token in sequence.logprobs]
            
            confidence_list.append([calculate_confidence(token) for token in sequence.logprobs])
            
            token_logprobs.append(logprob_list)

        results.append((generated_texts, token_logprobs, confidence_list))

    return results

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 初始化 vLLM 模型
    # 请将 'meta-llama/Llama-2-7b-chat-hf' 替换为你的模型路径或名称
    # 确保你已下载了模型权重并安装了必要的库
    # 'tensor_parallel_size' 参数可以让你使用多张 GPU
    try:
        model = LLM(model="/WORK/fit/alex/Kaisen/checkpoints/qwen/renew/global_step_200/huggingface", tensor_parallel_size=torch.cuda.device_count())
    except Exception as e:
        print(f"加载 VLLM 模型时出错：{e}")
        print("请确保你已下载模型权重并可以访问。")
        exit()

    # 2. 定义输入提示
    input_prompts = [
        "What is the capital of France?",
        "What is the largest planet in our solar system?"
    ]

    # 3. 生成并获取对数概率
    # 这将为每个提示生成 5 个样本
    results = generate_and_get_logprobs(
        model=model,
        input_text=input_prompts,
        max_new_tokens=300,
        sample_num=5,
        temperature=1.0,
        top_p=0.9
    )

    # 4. 打印结果
    for i, (texts, logprobs_list) in enumerate(results):
        print(f"--- 提示：'{input_prompts[i]}' ---")
        for j, (text, logprobs) in enumerate(zip(texts, logprobs_list)):
            print(f"样本 {j+1}:")
            print(f"  生成的文本: '{text.strip()}'")
            print(f"  Token 对数概率: {logprobs}")
        print("\n")