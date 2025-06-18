import ollama
from tqdm import tqdm
import pandas as pd
import csv
import re

model_list = ['qwen2', 'glm4','lawdemo1']
model = model_list[2]  # 选择模型
process_data = 500  # 要处理的法律文书数量，不超过9000
start_line = 0

def LLM_Process(model, sys_prom, usr_prom):
    messages = [
        {'role': 'user', 'content': usr_prom},
        {'role': 'system', 'content': sys_prom}
    ]
    options = {
        'temperature': 0.1
    }
    resp = ollama.chat(model, messages)
    try:
        out = resp['message']['content']
        return out
    except AttributeError:
        print("跳过处理。")
        return None

def truncate_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length]
    else:
        return text

inputdir = './data/test_set.csv'
output_csv = './output/' + model + '_processed_data.csv' 

sysP = '你是一名犯罪地理学家，请在我提供给你的法律文书中辨析犯罪发生的的地址并输出给我，如果有多条地址请只用换行符隔开，如果文书中没有提供详细的地址信息或你无法辨析，你只需要输出‘NaN’，，请不要回复除犯罪发生的地址之外的任何内容。谢谢你的配合！'
allin = pd.read_csv(inputdir)
keys = ['UUID', '正文']

end_line = start_line + process_data if start_line + process_data <= len(allin) else len(allin)
allin = allin.iloc[start_line:end_line]

results_df = pd.DataFrame(columns=['UUID', '正文', model])

with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['UUID', '正文', model])
    writer.writeheader() 

    for index, row in tqdm(allin.iterrows(), total=len(allin)):
        content2 = row[keys[1]]
        t = truncate_text(f"{content2}", 2200)
        out = LLM_Process(model, sysP, sysP + t)
        if out is not None:
            writer.writerow({'UUID': row[keys[0]], '正文': t, model: out})

print('process over')
