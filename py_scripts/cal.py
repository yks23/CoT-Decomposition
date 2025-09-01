from datasets import Dataset
import os
import json
from functools import reduce
import hydra
def compare_answers(pred, gt):
    return pred['answer'].replace(',','') == gt['answer'].replace(',','')

def search_answer(pred, gt):
    return gt['answer'].replace(',','') in pred['cot'].replace(',','')

def search_answer_last_line(pred, gt):
    return gt['answer'].replace(',','') in pred['cot'].split('\n')[-1].replace(',','')
    

def calculate_accuracy(predictions,references,function:callable=None):
    all_number = len(predictions)
    correct_number = reduce(lambda x,y:x+function(y[0],y[1]),zip(predictions,references),0)
    correct_mask = [function(pred, ref) for pred, ref in zip(predictions, references)]
    accuracy = correct_number / all_number
    return accuracy,correct_mask
@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg):
    data = []
    result_path = cfg.result_path
    dataset_name = cfg.dataset
    static_path = cfg.static_path
    result_dir = os.path.dirname(result_path)
    files = [f for f in os.listdir(result_dir) if f.startswith('result') and f.endswith('.json')]
    mask = None
    print(f"Result Directory: {result_dir}")
    print(f"Result files found: {files}")
    all_correct_s = 0
    all_number_s = 0
    for file in files:
        file_path = os.path.join(result_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        if dataset_name=='math':
            dataset = Dataset.from_parquet('./dataset/hendrycks_math/algebra/test-new.parquet')
            for i in range(len(data)):
                data[i]['gt_answer'] = dataset[i]['answer']
        
        predictions = [{'cot': d['pred_cot'], 'answer': d['pred_answer']} for d in data]
        references = [{'cot': d['gt_cot'], 'answer': d['gt_answer']} for d in data]
        accuracy,mask_new = calculate_accuracy(predictions, references, search_answer_last_line)
        for d in data:
            d['pred_answer'] = d['pred_cot'].split('\n')[-1]
        wrong_cases = [data[i] for i in range(len(data)) if not mask_new[i]]
        json.dump(wrong_cases, open(os.path.join(result_dir, f'wrong_cases_{file}'), 'w'), ensure_ascii=False, indent=4)
        if mask is None:
            mask = mask_new
        else:
            mask = [m or m_new for m, m_new in zip(mask, mask_new)]
        all_correct_s += sum(mask_new)
        all_number_s += len(mask_new)
        print("single:",sum(mask_new)/ len(mask_new))
        all_correct = reduce(lambda x,y:x+y, map(int,mask),0)
        print(f"Correct: {all_correct} / {len(mask)}")
        print(f"Accuracy of {file}: {all_correct}/{len(mask)} = {all_correct/len(mask):.2%}")
    print(f"Average Accuracy: {all_correct_s/all_number_s:.2%}")
    accuracy = all_correct / len(mask)
    with open(static_path, 'r') as f:
        static_data = json.load(f)
    now_data = {}
    now_data['path'] = result_path
    now_data['dataset'] = dataset_name
    now_data['accuracy'] = accuracy
    now_data['best@'] = len(files)
    static_data.append(now_data)
    with open(static_path, 'w') as f:
        json.dump(static_data, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()
        