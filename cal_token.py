from transformers import AutoTokenizer

def count_tokens(tokenizer_name: str, text: str) -> int:
    """
    计算输入字符串 text 的 token 数量，使用指定的 tokenizer。

    Args:
        tokenizer_name (str): Hugging Face 模型名称或本地 tokenizer 路径
        text (str): 待计算 token 数量的文本

    Returns:
        int: token 数量
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # 使用 tokenizer 编码文本为 token id
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


if __name__ == "__main__":
    # tokenizer_name = "/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"  # 你可以改成任意 tokenizer，例如 "facebook/opt-1.3b"
    # text = "**Solution Strategy:**\n1. Calculate the total cost of shorts.\n2. Calculate the total cost of pants.\n3. Calculate the total cost of shoes.\n4. Add the costs of shorts, pants, and shoes to find the total amount spent.\n\n"
    # num_tokens = count_tokens(tokenizer_name, "1. Let's denote the x-coordinates of the vertices of the triangle as a, b, and c.\n2. The sum of the x-coordinates of the vertices is given as a + b + c = √13.\n3. The midpoints of the sides of the triangle will have x-coordinates that are the averages of the x-coordinates of the vertices.\n4. We can find the x-coordinates of the midpoints as (a+b)/2, (a+c)/2, and (b+c)/2.\n5. The sum of the x-coordinates of the midpoints, S, is (a+b)/2 + (a+c)/2 + (b+c)/2.\n6. We can simplify S by combining like terms: S = (2a + 2b + 2c)/2.\n7. Simplifying further, we get S = a + b + c.\n8. We know that a + b + c = √13.\n9. Therefore, S = √13.\n10. We are asked to find S^2, which is (√13)^2.")
    # Model = 
    # print(f"文本的 token 数量: {num_tokens}")
    


