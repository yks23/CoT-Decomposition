import json
data_path_1 = './result/result1/result.json'
data_path_2 = './result/result4/result.json'
aw=[]
wa=[]
ww=[]
data_1=[]
data_2=[]
with open(data_path_1, 'r') as f:
    data_1 = json.load(f)
with open(data_path_2, 'r') as f:
    data_2 = json.load(f)
for d1,d2 in zip(data_1,data_2):
    cor1=False
    cor2=False
    try:
        d1['pred_answer']=float(d1['pred_answer'])
        d1['gt_answer']=float(d1['gt_answer'])
    except:
        pass
    if d1['pred_answer'] == d1['gt_answer']:
        cor1=True
    try:
        d2['pred_answer']=float(d2['pred_answer'])
        d2['gt_answer']=float(d2['gt_answer'])
    except:
        pass
    if d2['pred_answer'] == d2['gt_answer']:
        cor2=True
    if cor1 and (not cor2):
        aw.append({"question":d1['question'],'answer1':d1['pred_answer'],'answer2':d2['pred_answer']})
    if (not cor1) and cor2:
        wa.append({"question":d1['question'],'answer1':d1['pred_answer'],'answer2':d2['pred_answer']})
    if (not cor1) and (not cor2):
        ww.append({"question":d1['question'],'answer1':d1['pred_answer'],'answer2':d2['pred_answer']})
print(f"result1 独对: {len(aw)}")
print(f"result2 独对: {len(wa)}")
final_result = {}
final_result['1>2'] = aw
final_result['2>1'] = wa
final_result['1&2 wrong'] = ww
final_result['1_path']=data_path_1
final_result['2_path']=data_path_2
with open("./result/compare/result.json",'w') as f:
    json.dump(final_result, f, ensure_ascii=False, indent=4)
        