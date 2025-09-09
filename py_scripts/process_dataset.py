import pandas as pd
from datasets import Dataset,load_dataset
from util import analyze_token_counts_parallel as calculate_token
# from preverification.dataset import get_math
# from transformers import AutoTokenizer, AutoModelForCausalLM


"""
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("jaeyong2/Reason-Qwen3-14B-En")


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("SmallDoge/Qwen3-Long-Reasoning")

extract from them tonight.

"""
prompt= r"""Solve this Question: <question>\nthink step by step.Give your final answer in the boxed{}.Here is a guideline for you to solve this problem. Please follow it step by step.\n<guideline>"""

# --- 主程序 ---
if __name__ == "__main__":
    import json
    # dataset = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft.parquet")
    # print(len(dataset)
        #   )
    # print(dataset[0])
    # all_result=[]
    # for i in range(8):
    #     path = f"/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new.json_rank{i}.json"
    #     with open(path, 'r') as f:
    #         dataset = json.load(f)
    #     all_result += dataset
    # for i in range(8):
    #     path = f"/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new2.json_rank{i}.json"
    #     with open(path, 'r') as f:
    #         dataset = json.load(f)
    #     all_result += dataset
    # dataset = Dataset.from_list(all_result)
    # dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new.parquet')
    # dataset2  = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new2.parquet')
    # dataset = [data for data in dataset]
    # dataset2 = [data for data in dataset2]
    # merged = Dataset.from_list(dataset + dataset2)
    # print(len(merged))
    # print(merged[0])
    # merged.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw-sft-new.parquet')
    # files=[
    #     "../dataset/reasonmed/reasonmed.json_rank0.json",
    #     "../dataset/reasonmed/reasonmed.json_rank1.json",
    #     "../dataset/reasonmed/reasonmed-new.json_rank0.json",
    #     "../dataset/reasonmed/reasonmed-new.json_rank1.json",
    # ]
    
    # newdataset = []
    # for file in files:
    #     with open(file,'r') as f:
    #         dataset = json.load(f)
    #     newdataset += dataset
    # print(len(newdataset))
    # newdataset = Dataset.from_list(newdataset)
    newdataset=Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/reasonmed/reasonmed-sft.parquet')
    
    subset_50000 = newdataset.shuffle(seed=42).select(range(50000))
    
    subset_50000.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/reasonmed/reasonmed-sft-small.parquet')
    
    # with open("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/reasonmed/reasonmed.json_rank0.json",'r') as f:
        # dataset = json.load(f)
    # prompts = [data['question'] for data in dataset]
    # all_tokens = calculate_token(prompts, '/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B')
    # print(f"总token数: {all_tokens}")
    
    # response = [data['response'] for data in dataset]
    # all_tokens = calculate_token(response, '/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B')
    # print(f"总token数: {all_tokens}")
    
    # cot = [data['cot'] for data in dataset]
    # all_tokens = calculate_token(cot, '/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B')
    # print(f"总token数: {all_tokens}")
    
    # exp = [data['exploration'] for data in dataset]
    # all_tokens = calculate_token(exp, '/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B')
    # print(f"总token数: {all_tokens}")
    
    # result_files= [f"../evaluation/qw/new-short-270/olympiad_bench/static_0_rank{i}.json" for i in range(8)]
    # statis=[]
    # best_statis = []
    # all1 =0
    # all2 =0
    # import json
    # for result_file in result_files:
    #     with open(result_file, 'r') as f:
    #         results = json.load(f)
    #     statis.append(results['avg_success']*results['all_number'])
    #     best_statis.append(results['best_success']*results['all_number'])
    #     all1+=results['all_number']
    #     all2+=results['all_number']
    # print("avg:",sum(statis)/all1)
    # print("best:",sum(best_statis)/all2)
    # import json
    # result_file = "../evaluation/qw-dapo/dapo/result_0_merged.json"
    # with open(result_file, 'r') as f:
    #     results = json.load(f)
    
    
    # dataset = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/qwen-rl.parquet")
    # new_set = dataset.shuffle(seed=42).select(range(500))
    # new_set.to_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/qwen-rl-valid.parquet")
    # print(len(new_set))
    # print(dataset[25]['prompt'])
    # print(dataset[25]['exploration'])

    # new_dataset = []
    # for result, data in zip(results, dataset):
    #     data['exploration'] = [s['full_text'] for s  in result['samples']]
    #     new_dataset.append(data)
    # print(new_dataset[0])
    # new_dataset = Dataset.from_list(new_dataset)
    # new_dataset.to_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/qwen-rl.parquet")
    # from modelscope.msdatasets import MsDataset
    # ds =  MsDataset.load('AI-ModelScope/ReasonMed')
    # print(len(ds))
    # print(ds[0])
    # dataset = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qwen.parquet")
    # print(len(dataset))
    # from modelscope.msdatasets import MsDataset
    # ds =  MsDataset.load('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/medqa', subset_name='med_qa_en_source', split='test')
    
    # import json
    
    # from datasets import load_dataset
    # for subname in ['clinical_knowledge','college_biology','college_medicine','medical_genetics','professional_medicine','anatomy']:
    #     dataset = load_dataset("cais/mmlu", subname)['test']
    #     newdataset =  []
    # # dataset = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/pubmedqa/raw.parquet")
        
    #     for data in dataset:
    #         question = data['question']
    #         choice = data['choices']
    #         gt_answer = data['choices'][data['answer']]
    #         for i,opt in zip(['A','B','C','D'],choice):
    #             question += f"\n{i}. {opt}"
                
    #         newdataset.append({
    #             "question": question,
    #             "answer": {
    #                 "answer": gt_answer,
    #                 "answer_idx": ['A','B','C','D'][data['answer']],
    #             }})
    #     newdataset = Dataset.from_list(newdataset)
    #     newdataset.to_parquet(f'/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/mmlu/{subname}.parquet')
    #     print(len(newdataset))
    #     print(newdataset[0])
    # for data in dataset:
    #     question = data['question']
    #     for opt in ['opa','opb','opc','opd']:
    #         big_Case = opt.upper()[-1]
    #         question += f"\n{big_Case}. {data[opt]}"

    #     choice = ['a','b','c','d'][data['cop']]
        
    #     newdataset.append({
    #         "question": question,
    #         "answer": {
    #             "answer": data['op'+choice],
    #             "answer_idx": choice.upper(),  
    #         }})
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/pubmedqa/processed.parquet')
    # print(len(newdataset))
    # print(newdataset[0])
    # import json
    # newdataset = []
    # with open("../dataset/reasonmed/CoTMed.json",'r') as f:
    #     CoT_dataset = json.load(f)
    # with open("../dataset/reasonmed/ResponseMed.json",'r') as f:
    #     Response_dataset = json.load(f)
    # for data1,data2 in zip(CoT_dataset,Response_dataset):
    #     newdataset.append({
    #         "question":data1['instruction'],
    #         "cot":data1['output'],
    #         "response":data2['output']
    #     })
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/reasonmed/reasonmed.parquet')
    
    # for 
    # print(len(newdataset))
    
    
    
    # # print(dataset[0].keys())
    # # print(dataset[0])
    # # print(dataset[0])
    # dic ={}
    # newdataset = []
    # for data in dataset:
    #     newdataset.append(
    #         {
    #             "question": data['content'],
    #             "execution":data['text'],
    #         }
    #     )
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/qwen6/raw.parquet')
    # print(len(newdataset))
    # print(newdataset[0])
    
    # valid_set = newdataset.shuffle(seed=42).select(range(500))
    # valid_set.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/dapo/qwen-valid.parquet')
    
    # print(len(dic))
    # for i in range(8)
    # # result_path = '/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/dataset_all.json_rank0.json'
    
    
    # dataset = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/s1k/s1k-new.parquet")
    # dataset2 = Dataset.from_parquet("/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/s1k/s1k-new-test.parquet")
    # print(len(dataset))
    # # print(dataset[500]['gemini_exploration'])
    # # print(dataset[500]['deepseek_exploration'])
    # newdataset = []
    # for data,dd in zip(dataset,dataset2):
    #     newdataset.append(
    #         {
    #             'question': data['question'],
    #             'answer': data['answer'],
    #             'solution': data['solution'],
    #             'execution': dd['execution1'],
    #             'exploration': dd['exploration1'],
    #         }
    #     )
    #     newdataset.append(
    #         {
    #             'question': data['question'],
    #             'answer': data['answer'],
    #             'solution': data['solution'],
    #             'execution': data['execution2'],
    #             'exploration': data['exploration2'],
    #         }
    #     )
    # print(len(newdataset))
    # print(newdataset[0])
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/s1k/qwen-sft.parquet')
    # dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/s1k/llama-sft.parquet')
    # print(len(dataset))
    # print(dataset[0])
    # print(len(new
    #     for i in range(1,4):
    #         d={}
    #         d['answer'] = data['answer']
    #         d['question'] = data['question']
    #         d['exploration'] = data[f'exploration{i}']
    #         d['execution'] = data[f'execution{i}']
    #         d['reference'] = data[f'execution{i}']
    #         newdataset.append(d)
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/s1k/s1k-sft.parquet')
    # dname="zwhe99/DeepMath-103K"
    # dname = 'open-r1/OpenR1-Math-220k'
    # dataset = load_dataset(dname,split="train")
    # print(len(dataset))
    # print(dataset[0])
    
#     from transformers import AutoModelForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained('/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained('/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')

# # 关键一步：扩展 embedding 矩阵大小
    # model.resize_token_embeddings(len(tokenizer))
#     model.save_pretrained('/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')

    # ds = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k/test.parquet')
    # print(ds[0])
    
    # ds = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/math500/default.parquet') # *
    # print(ds[0])
    
    # ds = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/aime24/default.parquet')
    # print(ds[0])
    
    # ds = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/minerva/default.parquet') # *
    # print(ds[0])
    # print(ds[0])
    # print(len(ds))
    # print(ds[0])
    # import json
    # with open('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/rr.json', 'rb') as f:
    #     newdataset = json.load(f)
    # # newdataset = []
    # cnt_1=0
    # cnt_2=0
    # for data in newdataset:
    #     if data['answer'].replace(' ',"") in data['rollout'].split('\n')[-1]:
    #         cnt_1+=1
    #         data['correct']=True
    #     else:
    #         data['correct']=False
    #     # if data['answer'].replace(' ',"") in data['reference'].split('\n')[-1]: 
    #     #     cnt_2+=1
    # newdataset = Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/reg_dataset.parquet')
    # print(newdataset[0])
    # ds = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/reg_dataset.parquet')
    # print(len(ds))
    # print(ds[0])
    
    
    # print(cnt_1)
    # print(cnt_2)
        # data['question'] = data['question'].replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',"")
        # is_correct = 
        # newdataset.append(data)
    # dataset  = Dataset.from_list(newdataset)
    # dataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/mixed_dataset.parquet')
    # print(dataset[0])
    
    
    
    # import json
    # newdataset = []
    # for i in range(8):
    #     path = f"/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/dataset_all.json_rank{i}.json"
    #     with open(path, 'r') as f:
    #         newdataset += json.load(f)
    # cnt=0
    # for data in newdataset:
    #     if data['answer'].replace(' ',"") in data['execution'].split('\n')[-1]:
    #         cnt+=1
    # print(cnt)
    # print(cnt/len(newdataset))
    
    # process 1
    # dataset_all=Dataset.from_parquet('./dataset/math-algebra/test-new.parquet')
    
    # print(dataset_all[0])
    # newdataset = []
    # for d in dataset_all:
    #     d['answer'] = d['solution'].split('boxed{')[-1]
    #     last_right = d['answer'].rfind('}')
    #     if last_right != -1:
    #         d['answer'] = d['answer'][:last_right]
    #     print(d['answer'])
    #     newdataset.append(d)
    # newdataset = Dataset.from_list(newdataset)
    
    # newdataset.to_parquet('./dataset/math-algebra/test-new.parquet')
    
    
    # import json
    # newdataset = []
    # question_id = {}
    
    # for i in range(8):
    #     path = f"/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/dataset_all.json_rank{i}.json"
    #     with open(path, 'r') as f:
    #         newdataset += json.load(f)
        
    # for i,data in enumerate(newdataset):
    #         question_id[data['problem']] = i
    # final_dataset = []
    # for data in dataset_all:
    #     if data['problem'] in question_id:
    #         idx = question_id[data['problem']]
    #         data['execution'] = newdataset[idx]['execution']
    #     else:
    #         data['execution'] = ''
    #         continue
    #     data['question'] = data['problem']
    #     data.pop('problem')
    #     final_dataset.append(data)
    # newdataset = final_dataset
    # print(len(newdataset))
    # print(newdataset[0])   
    # dataset_all = Dataset.from_list(newdataset)
    # dataset_all.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/Final_SFT_1.parquet')
    
    
    #             # print(len(newdataset))
    # # print(newdataset[0])
    # newdataset=Dataset.from_list(newdataset)
    # newdataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/dataset_all.parquet')
    # print(len(newdataset))
    # newdataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/dataset_all.parquet')
    # for data in newdataset:
        # if data['answer']
    # 定义输入和输出路径
    # train_input_path = './data_processing/OpenR1-Math-220k_sft_formatted_v4.parquet'
    
    # train_output_path = './dataset/openrl/all_filtered.parquet'
    # dataset = Dataset.from_parquet(train_output_path)
    
    # small_subset_500= dataset.shuffle(seed=42).select(range(500))
    # small_subset_500.to_parquet('./dataset/openrl/sub_train.parquet')
    # dataset = []
    # import json
    # with open('/home/fit/alex/Kaisen.Yang/CoT Decomposition/data_processing/progress_v4.jsonl', 'r') as f:
    #     dataset = [json.loads(line) for line in f if line.strip()]
    # print(dataset[0])
    # final_dataset = []
    # for data in dataset:
    #     text = data['text']
    #     text= text.split('\n\nQ: ')[-1]
    #     question = text.split('\n\n<stochastic>')[0]+'\n<thinking>'
    #     answer='<stochastic>'+text.split('\n\n<stochastic>')[1]
    #     final_dataset.append({
    #         'question': question,
    #         'answer': answer
    #     })
    # print(f"处理后的数据集总行数: {len(final_dataset)}")
    # print(final_dataset[0]['question'])
    # print(final_dataset[0]['answer'])
    # parquet_dataset = Dataset.from_list(final_dataset)
    # parquet_dataset.to_parquet(train_output_path)
    
    
    # test_load =  Dataset.from_parquet(train_output_path)
    
    # sft_model = "/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    # rl_model_checkpoint =  '/WORK/fit/alex/Kaisen/checkpoints/RL-Larger/CoT/global_step_800/actor'

    # tokenizer = AutoTokenizer.from_pretrained(sft_model, trust_remote_code=True)
    # tokenizer.add_special_tokens({'additional_special_tokens': ['<EXPLORATION>','</EXPLORATION>','<EXECUTION>','</EXECUTION>']})
    # tokenizer.padding_side = "left"
    # tokenizer.save_pretrained(sft_model)
    
    # model = AutoModelForCausalLM.from_pretrained(sft_model)
    
    # model.resize_token_embeddings(len(tokenizer))
    # model.save_pretrained(sft_model)

# # 关键一步：扩展 embedding 矩阵大小
    # model.resize_token_embeddings(len(tokenizer))
#     model.save_pretrained('/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct')
    # import random
    # cnt=0
    # new_dataset = []
    # for data in dataset:
    #     answer = data['reward_model']['ground_truth']
    #     question = data['prompt'][0]['content'].split('the answer to the problem.\n\n')[-1].split('\n\nRemember to put your answer on ')[-1]
    #     solution = answer
    #     # answer = data['answer']
    #     new_dataset.append({
    #         'question': question,
    #         'answer': answer,
    #         'solution': solution
    #     })
    # new_dataset = Dataset.from_list(new_dataset)
    # os.makedirs('./dataset/dapo', exist_ok=True)
    # print(f"处理后的数据集总行数: {len(new_dataset)}")
    # new_dataset = new_dataset.to_parquet('./dataset/dapo/default.parquet')
    # new_dataset = Dataset.from_list(new_dataset)
    # new_dataset.to_parquet('./dataset/openrl/all_filtered.parquet')
    #     # print(encoded_question,encoded_answer)
    # print(cnt)
    
    # 随机 500 valid, 500 test, 剩下的 train
    # test_output_path = './dataset/openrl/test.parquet'
    # valid_output_path = './dataset/openrl/valid.parquet'
    # train_output_path = './dataset/openrl/train.parquet'
    # test_dataset = test_load.shuffle(seed=42).select(range(500))
    # valid_dataset = test_load.shuffle(seed=42).select(range(500, 1000))
    # train_dataset = test_load.shuffle(seed=42).select(range(1000, len(test_load)))
    # test_dataset.to_parquet(test_output_path)
    # valid_dataset.to_parquet(valid_output_path)
    # train_dataset.to_parquet(train_output_path)
    # # 处理并保存训练集
    # process_dataset_and_save(train_input_path, train_output_path)
    
    
    # # --- 展示处理后的一个示例 ---
    # print("\n--- 处理后的一个示例 ---")
    # processed_train_dataset = Dataset.from_parquet(train_output_path)
    
    # # 1000条test，1000条valid，剩下的
    # processed_train_dataset = processed_train_dataset.shuffle(seed=42).select(range(3000))
    
    # test_output_path = './dataset/openrl/test.parquet'
    # valid_output_path = './dataset/openrl/valid.parquet'
    # train_output_path = './dataset/openrl/train.parquet'
    
    # processed_train_dataset.select(range(1000)).to_parquet(test_output_path)
    # processed_train_dataset.select(range(1000, 2000)).to_parquet(valid_output_path)
    # processed_train_dataset.select(range(2000, len(processed_train_dataset))).to_parquet(train_output_path)
    
    
    # # 取第一行数据
    # example = processed_train_dataset[0]
    
    # print("\n问题 (question):")
    # print(example['question'])
    # print("\n回答 (answer):")
    # print(example['answer'])
    
    
    # # print(dataset_math[0]['problem']+'\n'+'<thinking>')

    # # tokenizer_name = '/home/fit/alex/.cache/modelscope/hub/models/deepseek-ai/deepseek-math-7b-instruct'
    # # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # # encoded_question = tokenizer.encode(example['question'], add_special_tokens=False)
    # # encoded_answer = tokenizer.encode(example['answer'], add_special_tokens=False)
    # # print(encoded_answer)
    # # print(encoded_question)
    # # print("\n所有数据集处理和保存完成。")
    # from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
    # ds = load_dataset("zwhe99/DeepMath-103K")
    # dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/ds_train.parquet')
    # print(dataset[0])
#     model_name = "Qwen/Qwen3-8B"

# # load the tokenizer and the model
#     tokenizer = AutoTokenizer.from_pretrained('/home/fit/alex/.cache/modelscope/hub/models/Qwen/Qwen3-8B')   
#     # tokenizer = AutoTokenizer.from_pretrained('/home/fit/alex/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
#     print(len(dataset))
#     new_dataset = []
#     for data  in dataset:
#         # data['prompt'][0]['content'] = data['prompt'][0]['content'].replace('\n\nRemember to put your answer on its own line after \"Answer:\".','').replace('Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n','')+'\n<thinking>'
#         # data['reward_model']['ground_truth'] = data['reward_model']['ground_truth']
       
#         # new_dataset.append(data)
#         # data['solution'] = data['solution'].replace('<stochastic>','<think>').replace('<answer>','</think>')
#         data['solution'] = data['solution'].replace('<|im_end|>','')
#         if tokenizer.encode(data['solution'], add_special_tokens=False, return_tensors='pt').size(1) > 1250:
#             continue
#         new_dataset.append(data)
#     print(new_dataset[0])
#     dataset = Dataset.from_list(new_dataset)
#     dataset.to_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/qw_train.parquet')
#     # test-500
#     # test = dataset.shuffle(seed=42).select(range(500))
#     # valid = dataset.shuffle(seed=42).select(range(500, 1000))
#     # train = dataset.shuffle(seed=42).select(range(1000, 11000))
#     # train.to_parquet('./dataset/dapo/train-quick.parquet')
#     # test.to_parquet('./dataset/dapo/test-short.parquet')
#     # valid.to_parquet('./dataset/dapo/valid-short.parquet')
#     # dataset.to
