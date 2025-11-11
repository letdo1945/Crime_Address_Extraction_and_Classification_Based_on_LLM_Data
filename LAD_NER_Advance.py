import ollama
from tqdm import tqdm
import pandas as pd
import csv
import re

model_list = ['qwen2.5', 'deepseek-r1', 'LEC_demo3', 'GLM4', 'qwen3']
model = model_list[2]  # 选择模型
process_data = 500  # 要处理的法律文书数量
start_line = 0
keyw = '正文'
def LLM_Process(model, sys_prom, usr_prom):
    messages = [
        {'role': 'user', 'content': usr_prom},
        {'role': "assistant", 'content': sys_prom}
    ]
    options = {
        # 'temperature': 0.1,
        "num_predict": 1000,
        # "repeat_penalty": 1.7,
        # "presence_penalty": 1.7,
        # "top_p": 0.9
    }

    resp = ollama.chat(model, messages,options=options)
    try:
        out = resp['message']['content']
        return out


    except AttributeError:
        # 可能是信息太长或有违规信息
        print("跳过处理。")
        return None

def truncate_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length]
    else:
        return text

def extract_json_content(text):
    """从文本中提取```json和```之间的内容（包含首尾标记）"""
    pattern = r'```json(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip() #group(1)则仅返回之间内容不包含首尾标记
    return ""

inputdir = './input/测试集/Test_Set.csv'
output_csv = './output/' + model + '_processed_data.csv'  # 输出CSV文件路径


sysP = '''
从裁判文书中提取犯罪信息并分类地址
请你分析这篇裁判文书内容,提取其中每起犯罪的时间、类型、地点信息，对提取的地址按照指定规则进行分类,并按指定格式输出JSON结果，不需要编写代码，也请勿输出除json结果外的其他信息
地址分类规则按照优先级顺序:
      "C7_模糊描述位置：含方位词（附近/旁/路边/对面/周围/周边/一带等）",
      "C2_门牌号地址：包含精确数字标识（号/幢/栋/室/单元/座等）且无方位词",
      "C4_交通枢纽：包含交通特征词（站/路口/交叉口/出入口等）",
      "C5_开放区域：包含开放空间特征词（停车场/广场/公园/工地等）",
      "C6_机构/设施/居住区：包含机构类型词（商场/小区/大学/医院等）",
      "C3_道路/路段：以道路特征词结尾（路/街/道/巷/弄等）",
      "C1_行政单元：纯行政区划（省/市/区/县/街道等）",
      "C8_其他地址：不符合以上任何特征"
输出格式: {
    "crime1": {
      "time": "犯罪时间",
      "address": "具体地址",
      "address_category": "地址类别(例如C1_行政单元)"
    }
    "crime2": {
      "time": "犯罪发生时间",
      "address": "具体地址",
      "address_category": "地址类别(例如C2_门牌号地址)"
    }
  }
'''
allin = pd.read_csv(inputdir)
keys = ['UUID', keyw]

# 确定处理数据的范围
end_line = start_line + process_data if start_line + process_data <= len(allin) else len(allin)
allin = allin.iloc[start_line:end_line]

# 准备一个空的DataFrame，用于实时存储处理结果
results_df = pd.DataFrame(columns=['UUID', keyw, 'LLM'])

# 打开CSV文件准备写入
with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['UUID', keyw, 'LLM'])
    writer.writeheader()  # 写入列名

    for index, row in tqdm(allin.iterrows(), total=len(allin)):
        content2 = row[keys[1]]
        out = LLM_Process(model, sysP, content2)
        if out is not None:
            if model == 'deepseek-r1':
                match = re.search(r'</think>(.*?)(?=<|$)', out, re.DOTALL)
                if match:
                    out = match.group(1).strip()
                else:
                    out = ""
            if model == 'qwen3':
                match = re.search(r'</think>(.*?)(?=<|$)', out, re.DOTALL)
                if match:
                    out = match.group(1).strip()
                else:
                    out = ""
            # print(out)
            writer.writerow({'UUID': row[keys[0]], keyw: content2, 'LLM': out}) # extract_json_content(out)
            # print(extract_json_content(out))

print('process over')
