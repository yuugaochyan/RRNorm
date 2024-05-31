import os
import json
from tqdm import tqdm
from core.configure import get_rr_config
config = get_rr_config()
# from tools.vec import Vec_Search_Controller
# os.environ['OPENAI_API_BASE'] = "https://api.emabc.xyz/v1"
os.environ['OPENAI_API_BASE'] = config["OPENAI_API_BASE"]

# api_key = "sk-Tmbz7yyn96MMVY5uC4Ca6d4878F6496aB9686f9b94B78793"
api_key = config["API_KEY"]

import openai
openai.api_key=api_key
def generate(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    message = response["choices"][0]["message"]["content"]
    return message.encode('utf-8').decode('utf-8')

def make_prompt(q):
    prompt=""
    prompt+="你要完成一个组成成分识别任务。以最细粒度按照“发病内容”、“发病范围”、“操作内容”、“修饰词”、“分割词”、“无效内容”识别。\n\n"
    prompt+="以下是各个识别类别的含义：\n"
    prompt+="发病内容：可能存在的发病范围的症状、病变、状态等，以及“心脏病”、“高血压”等固定疾病名称。\n"
    prompt+="发病范围：产生病变的解刨部位。\n"
    prompt+="操作内容：放疗、化疗等治疗手段，或胃镜检查等检查手段。\n"
    prompt+="修饰词：描述发病内容程度、性质的词。或者描述发病范围的方位词\n"
    prompt+="分割词：“伴”、“并”等分割词、分隔符\n"
    prompt+="无效内容：“待查”、“怀疑”等无意义描述，或检查结果、病因等非疾病、检查内容，或发病范围的方位、区域。\n"
    prompt+="\n按照输入的内容逐个识别，只以JSON格式输出，例如：\n"
    prompt+=json.dumps([
{"原词":"恶性","类别":"修饰词"},
{"原词":"卵巢","类别":"发病范围"},
{"原词":"癌","类别":"发病内容"},
{"原词":"化疗后","类别":"操作内容"},
{"原词":"，伴","类别":"分割词"},
{"原词":"尿频","类别":"发病内容"},
{"原词":"可能性大","类别":"无效内容"},
],indent=2,ensure_ascii=False)
    prompt+="\n注意：必须按照输出的格式并且仅仅以JSON list形式输出，不要额外描述，不要缺少原词也不要增加额外的词语。\n"
    prompt+="输入："+q
    return prompt

def seg_gpt(mention,std):

    flag = True
    while flag:
        try:
            answer = json.loads(generate([
                {"role":"system","content":"你是一个医学助手"}, 
                {"role":"user","content":make_prompt(mention)}]))
            if not isinstance(answer,list):
                raise TypeError
            for part in answer:
                if "原词" not in part.keys() or "类别" not in part.keys():
                    raise TypeError
            flag = False
        except:
            flag = True
            print("failed retry")
    tmp = {
        "text":mention,
        "normalized_result":std,
        "gpt_sub":answer
    }
    return tmp

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True
def rule_compose_stage_1(o_text,label_result):
    result_list = []
    start = 0
    for index,item in enumerate(label_result):
        word = item["原词"]
        label = item["类别"]
        if label == "发病内容":
            pre_text = ""
            for pre in label_result[start:index+1]:
                if pre["类别"]=="修饰词":
                    pre_text+=pre["原词"]
            text = pre_text+word
            pre_text = ""
            for pre in label_result[start:index]:
                if pre["类别"]=="发病范围":
                    pre_text+= pre["原词"]
                    result_list.append(pre["原词"]+text)
            if pre_text == "":
                now = index - 1
                while now >= 0:
                    if label_result[now]["类别"]=="发病范围":
                        if index - now > 2:
                            now = -1
                            break
                        else:
                            result_list.append(label_result[now]["原词"]+text)
                            break
                    else:
                        now-=1
                if now == -1:
                    result_list.append(text)
            else:
                result_list.append(pre_text+text)
            start = index
        elif label == "操作内容":
            pre_text = ""
            text = word
            for pre in label_result[start:index]:
                if pre["类别"]=="发病内容":
                    pre_text=pre["原词"]
            # for pre in label_result[start:index]:
            #     if pre["类别"]=="修饰词":
            #         pre_text+=pre["原词"]
            result_list.append(pre_text+text)
            for pre in label_result[start:index]:
                if pre["类别"]=="发病范围":
                    pre_text= pre["原词"]
                    result_list.append(pre_text+text)
            # for pre in label_result[start:index]:
            #     if pre["类别"]=="修饰词":
            #         pre_text+=pre["原词"]
            # result_list.append(pre_text+text)
            start = index
        elif label == "分割词":
            if index+1 == len(label_result):
                result_list[-1]=result_list[-1]+word
                continue
            elif "致" in word:
                result_list.append(label_result[index-1]["原词"]+"所致的"+label_result[index+1]["原词"])
            elif label_result[index-1]["类别"]=="发病内容" and label_result[index+1]["类别"]=="发病内容":
                if is_all_chinese(word):
                    reverse_label_list = label_result[:start].copy()
                    if len(reverse_label_list)==0:
                        start = index
                    else:
                        reverse_label_list.reverse()
                        now = 0
                        while now < len(reverse_label_list) and reverse_label_list[now]["类别"] == "发病范围":
                            now+=1
                        start -=now
                else:
                    start = index
            elif label_result[index-1]["类别"]=="发病内容" and label_result[index+1]["类别"]=="发病范围":
                start = index - 1
            elif label_result[index-1]["类别"]=="发病范围" and label_result[index+1]["类别"]=="发病范围":
                start = index-1
        elif label == "修饰词":
            if len(label_result) == 1:
                result_list.append(word)
            elif index+1 == len(label_result):
                if len(result_list) == 0:
                    result_list.append(word)
                else:
                    result_list[-1]=result_list[-1]+word
        elif label == "发病范围":
            if index+1 == len(label_result):
                result_list.append(word)
    if len(result_list) == 0:
        result_list.append(o_text.replace("ca","癌").replace("Ca","癌").replace("CA","癌"))
    dedu_list = list(set(result_list))
    dedu_list.sort(key = result_list.index)
    return dedu_list
def rule_compose_stage_2(text,res_list):
    cancer_type_list =["腺癌","粘液腺癌","鳞癌","腺鳞癌"]
    cancer_xian = ["中肾腺癌","粘液腺癌","乳腺癌","胰腺癌","管状腺癌","甲状腺癌"]
    flag = True
    cancer_state_list = ["转移","复发"]
    for index,res in enumerate(res_list):
        res_list[index] = res.replace("ca","癌").replace("Ca","癌").replace("CA","癌")
        res_list[index] = res.replace("mt","恶性肿瘤").replace("Mt","恶性肿瘤").replace("MT","恶性肿瘤")
    if "腺癌" in text and "腺癌" not in res_list:
        if "癌" in res_list:
            res_list.remove("癌")
        res_list.append("腺癌")
        flag = False
        if "转移性" in text and "转移性腺癌" not in res_list:
            res_list.append("转移性腺癌")
    if "粘液腺癌" in text and "腺癌" in res_list:
        if "癌" in res_list:
            res_list.remove("癌")
        # if "腺癌" in res_list:
        res_list.remove("腺癌")
        if "粘液腺癌" not in res_list:
            res_list.append("粘液腺癌")
    if "乳腺癌" in text and "腺癌" in res_list:
        if "癌" not in res_list:
            res_list.append("癌")
        # if "腺癌" in res_list:
        res_list.remove("腺癌")
        for index,res in enumerate(res_list):
            if "乳腺癌" in res:
                res_list[index] = "乳腺癌"
        if "乳腺癌" not in res_list:
            res_list.append("乳腺癌")
    if "胰腺癌" in text and "腺癌" in res_list:
        if "癌" not in res_list:
            res_list.append("癌")
        # if "腺癌" in res_list:
        res_list.remove("腺癌")
        if "胰腺癌" not in res_list:
            res_list.append("胰腺癌")
    if "管状腺癌" in text and "腺癌" in res_list:
        if "癌" in res_list:
            res_list.remove("癌")
        res_list.remove("腺癌")
        if "管状腺癌" not in res_list:
            res_list.append("管状腺癌")
    if "甲状腺癌" in text and "腺癌" in res_list:
        if "癌" in res_list:
            res_list.remove("癌")
        # if "腺癌" in res_list:
        res_list.remove("腺癌")
        if "甲状腺癌" not in res_list:
            res_list.append("甲状腺癌")
    if flag  and "鳞癌" in text and "鳞癌" not in res_list:
        res_list.append("鳞状细胞癌")
    if flag and "癌" in text and "癌" not in res_list:
        res_list.append("癌")
        if "转移性" in text and "转移性癌" not in res_list:
            res_list.append("转移性癌")
    if "复发" in text and "癌" in text or ("复发" in text and "瘤" in text) and "恶性肿瘤复发" not in res_list:
        res_list.append("恶性肿瘤复发")
    for index,res in enumerate(res_list):
        flag = True
        if "癌" in res and len(res)>1:
            for cancer_type in cancer_type_list:
                if cancer_type in res:
                    flag = False
            if flag:
                res_list[index] = res.replace("癌","恶性肿瘤")
                res_list.append("癌")
    if "孕" in text and "不孕" not in text or "妊" in text:
        if "已产" in text:
            res_list.append("分娩")
        else:
            res_list.append("妊娠")
    dedu_list = list(set(res_list))
    dedu_list.sort(key = res_list.index)
    return dedu_list

# corpus = []
with open("data/knowledge_base/code.txt", "r", encoding='utf-8') as f:
    line = f.readline()
    while line:
        corpus.append(line.strip().split('\t')[1])
        line = f.readline()
vec_engine = Vec_Search_Controller(model_name_or_path=config["FILTER_METHOD"], init_kb=corpus)

def rule_compose_stage_3(text,res_list):
    micro_list = []
    scores = vec_engine.search_with_score(res_list,topN=1)
    for res,item in zip(res_list,scores):
        # top1=item[0][0]
        score=item[0][1]
        if score>0.7:
            micro_list.append(res)
    return micro_list