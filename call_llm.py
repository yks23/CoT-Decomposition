import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import tqdm
import json
import os
import torch
import re
from transformers import LogitsProcessor
import hydra
from omegaconf import OmegaConf
from preverification.config import name2path

class TemperatureScheduler(LogitsProcessor):
    """
    动态调整生成温度：
    - 前 first_tokens 使用 first_temp
    - 后续使用 second_temp
    - 遇到 special_seg 后切换温度，使用 flag 避免重复解码
    """
    def __init__(self, first_tokens, first_temp, second_temp, tokenizer, special_seg="ROLL OUT", check_after=100):
        self.first_tokens = first_tokens
        self.first_temp = first_temp
        self.second_temp = second_temp
        self.special_seg = special_seg
        self.tokenizer = tokenizer
        self.check_after = check_after
        self.triggered_flags = None  # batch_size 大小的 flag，在生成前初始化

    def init_flags(self, batch_size):
        # 初始化每个样本的 flag
        self.triggered_flags = [False] * batch_size

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 如果 trigger_flags 还没初始化
        if self.triggered_flags is None or len(self.triggered_flags) != batch_size:
            self.init_flags(batch_size)

        for i in range(batch_size):
            temp = None

            # 还没触发 special_seg
            if not self.triggered_flags[i]:
                # 超过指定 token 后才检查
                if seq_len > self.check_after:
                    decoded_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    decoded_text = decoded_text.split("assistant")[-1]
                    if self.special_seg in decoded_text:
                        self.triggered_flags[i] = True  # 设置 flag

            # 根据 flag 和 first_tokens 决定温度
            if self.triggered_flags[i] or seq_len > self.first_tokens:
                temp = self.second_temp
            else:
                temp = self.first_temp

            # 调整 logits
            if temp == 0:
                argmax_token = torch.argmax(scores[i])
                new_scores = torch.full_like(scores[i], -1e9)
                new_scores[argmax_token] = 1e9
                scores[i] = new_scores
            elif temp != 1.0:
                scores[i] = scores[i] / temp

        return scores


class APIMathSolver:
    def __init__(self,mode='with',max_new_tokens=1300,prompt=""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        if mode == "without":
            self.sys_prompt = "Please reason step by step, and put your final answer at the lastline."
        elif mode == "with":
            self.sys_prompt = prompt
    def solve_batch(self,questions):
        prompts = [self.sys_prompt]*len(questions)
        response = multi_conservation(
            system_prompt=prompts,
            user_input=questions,
            need_json=False,
            show_progress=True)
        
        results = []
        for full_text in response:
            boxed_matches = re.findall(r"\\boxed\{(.+?)\}", full_text, re.DOTALL)
            final_answer = boxed_matches[-1].strip() if boxed_matches else None
            results.append((full_text, final_answer))
        return results
    
class QwenMathSolver:
    def __init__(self, model_path="", device="cuda", mode="with", max_new_tokens=1300,prompt=""):
        self.device = device
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"  # 关键：左侧padding，防止警告
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=None,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        if mode == "without":
            self.sys_prompt = ""
        elif mode == "with":
            self.sys_prompt = prompt
        else:
            raise ValueError("mode 必须是 'with' 或 'without'")
        
    @torch.no_grad()
    def solve_batch_tp(self, questions, temp_strategy=0):
        """
        temp_strategy:
        0 -> temperature=0
        1 -> 前30token=1.3, 后续=0
        2 -> 前30token=0, 后续=1.3
        """
        # 准备 batch
        messages_list = [
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer at the lastline."},
                {"role": "user", "content": self.sys_prompt + question}
            ]
            for question in questions
        ]
        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        batch_input = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

        # 存储生成结果（按原始顺序）
        final_texts = [""] * len(questions)

        # 动态存活索引
        alive_indices = list(range(len(questions)))
        generated_ids = batch_input.input_ids.clone()

        for idx in range(self.max_new_tokens):
            # forward 仅对当前存活样本
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=(generated_ids != self.tokenizer.pad_token_id)
            )
            logits = outputs.logits[:, -1, :]

            # 温度策略
            if temp_strategy == 0:
                temperature = 0.0
            elif temp_strategy == 1:
                temperature = 1.3 if idx < 30 else 0.0
            elif temp_strategy == 2:
                temperature = 0.0 if idx < 30 else 1.3
            elif temp_strategy == 3:
                temperature = 1.0
            else:
                raise ValueError("temp_strategy 必须是 0, 1, 2")

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # 找到生成 EOS 的样本
            eos_mask = (next_token.squeeze(1) == self.tokenizer.eos_token_id)
            print(f"step {idx+1}, alive samples: {len(alive_indices)}, just finished: {eos_mask.sum().item()}temperature: {temperature}")
            if eos_mask.any():
                # 解码结束的样本，并记录到 final_texts
                for i in torch.nonzero(eos_mask, as_tuple=False).view(-1).tolist():
                    orig_idx = alive_indices[i]
                    text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    final_texts[orig_idx] = text
                # 去掉结束的样本
                keep_mask = ~eos_mask
                generated_ids = generated_ids[keep_mask]
                alive_indices = [alive_indices[i] for i in range(len(alive_indices)) if keep_mask[i].item()]

            # 所有样本都结束
            if not alive_indices:
                break

        # 如果还有没结束的样本，最后一起解码
        if alive_indices:
            for i, orig_idx in enumerate(alive_indices):
                text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                final_texts[orig_idx] = text

        # 提取答案
        results = []
        for text in final_texts:
            boxed_matches = re.findall(r"\\boxed\{(.+?)\}", text, re.DOTALL)
            final_answer = boxed_matches[-1].strip() if boxed_matches else text.split("<answer>\n")[-1].strip()
            results.append((text, final_answer))

        return results


    @torch.no_grad()
    def solve_batch_segmented(self, questions, first_temp=1.0, second_temp=0.0, first_tokens=50, split_token="|"):
        """
        一次性生成，前 first_tokens 个 token 使用 first_temp，后续使用 second_temp。
        在第一段和第二段交界处插入 split_token。
        """
        # 构造批量输入
        messages_list = [
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer at the lastline."},
                {"role": "user", "content": self.sys_prompt + question}
            ]
            for question in questions
        ]
        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        batch_input = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

        # 设置分段温度
        logits_processor = [TemperatureScheduler(first_tokens=first_tokens,
                                                first_temp=first_temp,
                                                second_temp=second_temp,
                                                tokenizer=self.tokenizer)]

        # === 一次性生成全部 token ===
        full_gen = self.model.generate(
            **batch_input,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=logits_processor
        )

        # 批量解码
        full_texts = self.tokenizer.batch_decode(full_gen, skip_special_tokens=True)

        # 插入交界符 split_token
        texts_with_split = []
        for text in full_texts:
            if len(text) > first_tokens:
                text_with_split = text[:first_tokens] + split_token + text[first_tokens:]
            else:
                text_with_split = text + split_token
            texts_with_split.append(text_with_split)

        # 提取答案
        results = []
        for text in texts_with_split:
            final_answer = text.split("<answer>")[-1]
            results.append((text, final_answer))

        return results

    @torch.no_grad()
    def solve_batch_official(self, questions, temperature=0.0,change_input=True):
            """
            用 Hugging Face 官方 generate 实现批量推理
            :param questions: list[str] 问题列表
            :param temperature: float 统一温度 (0.0 = greedy)
            """
            # 构造批量输入
            if change_input:
                messages_list = [
                    [
                        {"role": "system", "content": "Please reason step by step, and put your final answer at the lastline."},
                        {"role": "user", "content": self.sys_prompt + question}
                    ]
                    for question in questions
                ]
                texts = [
                    self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in messages_list
                ]
                batch_input = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

            else:
                batch_input = self.tokenizer(questions, return_tensors="pt", padding=True).to(self.device)
            # 官方并行生成
            generated_ids = self.model.generate(
                **batch_input,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),  # 如果 temperature > 0 走采样，否则 greedy
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 批量解码
            full_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 提取答案
            results = []
            for text in full_texts:
                boxed_matches = re.findall(r"\\boxed\{(.+?)\}", text, re.DOTALL)
                final_answer = boxed_matches[-1].strip() if boxed_matches else text.split("<answer>\n")[-1].strip()
                results.append((text, final_answer))

            return results


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg):
    use_api = cfg.use_api
    if use_api:
        from multiprocessing import freeze_support
        freeze_support() # 多线程必须使用！！
        from preverification.call_api import multi_conservation
    seed = cfg.number
    torch.manual_seed(seed)
    use_prompt = cfg.use_prompt
    batchsize = cfg.batch_size
    output_path = cfg.output_path
    number = cfg.number
    tp1 = cfg.tp1
    model = cfg.model_name
    tp2 = cfg.tp2
    dataset_name = cfg.dataset
    from preverification.dataset import split_answer, get_gsm8k, get_math,get_modified_gsm8k
    if use_prompt:
        mode = 'with'
        print(f"Using prompt")
    else:
        mode = 'without'
        print(f"Not using prompt")

    if dataset_name == 'gsm8k':
        dataset = get_gsm8k()['test']
    elif dataset_name == 'math':   
        dataset = get_math()['test']
    elif dataset_name == 'modified_gsm8k':
        dataset = get_modified_gsm8k()
    with open(cfg.prompt_path, 'r') as f:
        prompt = f.read()
    if not use_api:
        solver = QwenMathSolver(model_path=name2path[model],mode=mode,prompt=prompt)
    else:
        solver = APIMathSolver(mode=mode)
        
    time_start = time.time()
    batchsize = batchsize
    
    results = []
    bar = tqdm.tqdm(total=len(dataset), desc="Processing questions")
    if output_path is not None:
        dir_name = output_path
    os.makedirs(dir_name, exist_ok=True)
    for i in range(0, len(dataset), batchsize):
            if i + batchsize > len(dataset):
                batchsize = len(dataset) - i
            bar.update(batchsize)
            if dataset_name == 'gsm8k':
                questions = dataset[i:i + batchsize]['question']
                gt_answers = dataset[i:i + batchsize]['answer']
            elif dataset_name == 'math':
                questions = dataset[i:i + batchsize]['problem']
                gt_answers = dataset[i:i + batchsize]['solution']
            elif dataset_name == 'modified_gsm8k':
                questions = dataset[i:i + batchsize]['question']
                gt_answers = dataset[i:i + batchsize]['answer']
            first_token=50 if dataset_name != 'math' else 200
            pred_result = solver.solve_batch_segmented(questions,first_temp=tp1,second_temp=tp2,first_tokens=first_token)
                
            for j, (full_text,final_answer) in enumerate(pred_result):
                gt_cot, gt_answer = split_answer(gt_answers[j],type = dataset_name)
                meta ={}
                meta['question'] = questions[j]
                meta['pred_cot'] = full_text.split(questions[j])[-1]
                meta['gt_cot'] = gt_cot
                meta['pred_answer'] = final_answer
                meta['gt_answer'] = gt_answer
                results.append(meta)
            print("一批的预测结果已经完成。")
            
            with open(os.path.join(dir_name,f'result_{number}.json'), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
    time_end = time.time()
    print(f"总耗时: {time_end - time_start:.2f}秒")
    print(f"平均每题耗时: {(time_end - time_start) / len(questions):.2f}秒")
if __name__ == "__main__":
    main()