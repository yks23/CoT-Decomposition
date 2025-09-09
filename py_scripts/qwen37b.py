from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
# model_name = "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/checkpoints/CoT/qwen37b/global_step_600/huggingface"
# load the tokenizer and the model
import torch
torch.cuda.device
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.to("cuda:0")

# prepare the model input
# prompt = "Give me a short introduction to large language model."
prompt = "Find the solution to the problem: If x^2 +x + 1 = 2, what is the value of x?"  # 输入文本

messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)


print(text)
tokenizer.padding_side = 'left'
model_inputs = tokenizer(text, return_tensors="pt",max_length=200,padding='max_length').to(model.device)
# conduct text completion
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2000
)
print(generated_ids.shape)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=False).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)