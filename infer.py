import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from preverification.dataset import get_modified_gsm8k,get_math
class SimpleModelLoader:
    def __init__(self, model_path="", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)

    def generate_answer(self, question):
        
        # Tokenize the input question
        message= {
            "role": "user",
            "content": question
        }
        text = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # Generate output from the model
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], max_length=1000, num_return_sequences=1,do_sample=False)

        # Decode and return the generated text
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return answer


if __name__ == "__main__":
    # Set the model path (either local path or Hugging Face model hub name)
    model_path= "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    # model_path ='/home/fit/alex/.cache/modelscope/hub/models/deepseek-ai/deepseek-math-7b-instruct'
    # Initialize the model loader
    model_loader = SimpleModelLoader(model_path=model_path, device="cuda")
    dataset = get_math()['test']
    # Get the model's answer
    for i in range(10,20):
        question = dataset[i]['problem']+'\n'+'<thinking>'
        print(f"Question {i+1}: {question}")
         # Generate the answer
        answer = model_loader.generate_answer(question)
        print(f"Answer {i+1}: {answer}")
        print(f"Ground Truth: {dataset[i]['solution']}")
        with open('test.txt', 'a') as f:
            f.write(f"Question {i+1}: {question}\n")
            f.write(f"Answer {i+1}: {answer}\n")
            f.write(f"Ground Truth: {dataset[i]['solution']}\n\n")
        # print(answer.split("<answer>")[-1].strip(),'-----', dataset[i]['answer'].split("<answer>")[-1].strip())
