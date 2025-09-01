import json
import os 
if __name__ == "__main__":
    files = [
        "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/verl/trainer/evaluation/openrl/raw-100",
        "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/verl/trainer/evaluation/openrl/full-100",
        "/home/fit/alex/Kaisen.Yang/CoT Decomposition/verl/verl/trainer/evaluation/openrl/llama"
    ]
    static=[]
    result = []
    for file in files:
        result_file = os.path.join(file,'result_0.json')
        with open(result_file,'r') as f:
            result.append(json.load(f))
        static_file = os.path.join(file,'static_0.json')
        with open(static_file,'r') as f:
            static.append(json.load(f))
    for i in range(len(static[0]['vector'])):
        all_correct = True
        for j in range(len(static)):
            if not static[j]['vector'][i]:
                all_correct = False
        if all_correct:
            continue
        else:
            # print(f"problem {i}")
            print("========")
            # print(result[0][i]["question"])
            for j in range(len(static)):
                print(f"model {j} : ")
                print("---------------------------------------------------")
                # print(result[j][i]["deterministic"].replace("<|eot_id|>",""))
                print(f"correct: {static[j]['vector'][i]}")
        next = input("next?")