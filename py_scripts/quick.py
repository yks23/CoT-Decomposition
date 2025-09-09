# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('open-r1/DAPO-Math-17k-Processed', subset_name='all', split='train')
# # dapo
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('opencompass/AIME2025', subset_name='AIME2025-I', split='test')
# # AIME 2025 or AIME2025-II
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/Maxwell-Jia-AIME_2024', subset_name='default', split='train')
# # AIME 2024 
# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('knoveleng/AMC-23', subset_name='default', split='train')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/MATH-500', subset_name='default', split='test')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('knoveleng/Minerva-Math', subset_name='default', split='train')

# # from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('AI-ModelScope/OlympiadBench')
# from datasets import load_dataset

# ds = load_dataset("HuggingFaceH4/aime_2024")
# print(ds['train'][0])
import re
from reward import normalize_final_answer
def get_max_step(text: str) -> int:
    """
    提取文本中的序号步骤，返回最大序号（支持任意位数字）

    Args:
        text (str): 输入文本

    Returns:
        int: 文本中最大的序号，如果没有找到返回0
    """
    # 匹配任意位数字序号，格式如 1. 或 2)
    matches = re.findall(r'\b(\d+)[\.\)]', text)
    numbers = [int(num) for num in matches]
    return max(numbers) if numbers else 0

def extract_boxed_content(s: str) -> list[str]:
    """
    Parses a string and extracts all content enclosed by \\boxed{}.

    Args:
        s: The input string.

    Returns:
        A list of strings, where each string is the content found inside a \\boxed{} block.
    """
    results = []
    # Use a counter to track the nesting level of curly braces
    brace_level = 0
    # Store the starting index of a \\boxed{} block
    start_index = -1
    
    i = 0
    while i < len(s):
        # Look for the start of a new \\boxed{} block
        if s[i:i+6] == "boxed{":
            # If we are not already inside a block, set the start index
            if brace_level == 0:
                start_index = i + 6
            # Increment the brace level
            brace_level += 1
            # Move the index past the opening part of \\boxed{}
            i += 6
            continue
        
        # If we are inside a \\boxed{} block (brace_level > 0)
        if brace_level > 0:
            if s[i] == "{":
                brace_level += 1
            elif s[i] == "}":
                brace_level -= 1
                # If brace_level drops to 0, we have found a complete \\boxed{} block
                if brace_level == 0:
                    content = s[start_index:i]
                    results.append(content)
                    start_index = -1  # Reset start index
        
        i += 1
            
    return results

# file_path = "/home/fit/alex/Kaisen.Yang/CoT Decomposition/evaluation/qw/rlnew/300/math500/result_0_rank0.json"
# import json
# with open(file_path,'r') as f:
#     results = json.load(f)

# finished = 0
# correct = 0
# all = 0
# for res in results:
#     for sample in res['samples']:
#         if len(extract_boxed_content(sample['full_text'])):
#             finished += 1
#             if sample['success']:
#                 correct += 1
#         all += 1
# print(f"finished: {finished}, correct: {correct}, all: {all}, finish_rate: {finished/all}, finish_and_correct_rate: {correct/finished if finished>0 else 0}")
from datasets import Dataset
import json
# dataset  = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new.parquet")

with open('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new.json_rank7.json','r') as f:
    dataset = json.load(f)
cal={}
for data in dataset:
    # steps = get_max_step(data['exploration'])
    # if steps<0 or steps>10:
    #     continue
    # if steps not in cal:
    #     cal[steps] = 0
    # cal[steps] += 1
    # if 'Conclude' in data['exploration'] or 'conclude' in data['exploration'] or 'CONCLUDE' in data['exploration']:
    #     cnt_conclude = cal.get('conclude',0)
    #     cal['conclude'] = cnt_conclude + 1
    #     print("found conclude:", data['exploration'])
    for boxed in extract_boxed_content(data['execution']):
        gt_answer,p = normalize_final_answer(data['answer'])
        pred_answer,pp = normalize_final_answer(boxed)
        if gt_answer == pred_answer or (p is not None and pp is not None and abs(p - pp) < 1e-3):
            cal['execution'] = cal.get('execution',0) + 1
    for boxed in extract_boxed_content(data['re_execution']):
        gt_answer,p = normalize_final_answer(data['answer'])
        pred_answer,pp = normalize_final_answer(boxed)
        if gt_answer == pred_answer or (p is not None and pp is not None and abs(p - pp) < 1e-3):
            cal['re_execution'] = cal.get('re_execution',0) + 1
    for boxed in extract_boxed_content(data['short-re_execution']):
        gt_answer,p = normalize_final_answer(data['answer'])
        pred_answer,pp = normalize_final_answer(boxed)
        if gt_answer == pred_answer or (p is not None and pp is not None and abs(p - pp) < 1e-3):
            cal['short-re_execution'] = cal.get('short-re_execution',0) + 1
print(cal)