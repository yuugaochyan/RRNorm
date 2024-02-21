import json
import pandas as pd
import datasets
from transformers import TrainerCallback, AutoTokenizer
import random
import re
import torch
from core.configure import get_recall_config
from core.datasets.common import *
args,_ = get_recall_config()
tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
normalized_words_set = readCorpus(args.corpus_path+"code.txt")

def read_recall_data(filename):
    """加载数据
    格式：[(文本1, 文本2, 标签id)]
    """
    D = []
    with open(filename) as f:
        all_data = json.loads(f.read())
        for i, l in enumerate(all_data):
            text = re.sub('\[.*?\]','',l['text'])
            normalized_result = l.get('normalized_result', '')
            if len(normalized_result) > 0:
                split_result = normalized_result.split("##")
                # 跳过原始一对多数据 只留下一对一
                if len(split_result)>1:
                    continue
                normalized_result = []
                for item in split_result:
                    item = normal_word(item)
                    if item:
                        normalized_result.append(item)
                        
                for item in normalized_result:
                    if item not in normalized_words_set:
                        normalized_words_set.add(item)
                        
            else:
                normalized_result = []
            if len(normalized_result) > 0:
                # for item in normalized_result:
                D.append((text, normalized_result))
    return D

def select_random_items(lst,n=None):
    if n is None:
        num_to_select = random.randint(1, len(list))
    else:
        num_to_select = n
    selected_items = random.sample(lst, num_to_select)
    return selected_items

def preprocess_function(examples):
    inputs = tokenizer(examples['input'],padding='max_length',max_length=48,return_tensors='pt',truncation=True)                             
    return inputs

def get_triplet_dataset(data_list,standard_name_list,neg_num):
    input_data = []
    golden_list = []
    for item in data_list:
        item_neg_list = select_random_items(standard_name_list,neg_num*5)
        item_neg_list = [i for i in item_neg_list if i not in item[1]][:neg_num]
        
        for pos in item[1]:
            input_data.append([item[0],pos]+item_neg_list)
            golden_list.append(torch.tensor([1,0,0,0,0]))
    
    df = pd.DataFrame({'input':input_data})
    dataset = datasets.Dataset.from_pandas(df)
    encoded_dataset = dataset.map(preprocess_function,num_proc=8)
    return encoded_dataset

def get_pair_dataset(data_list,standard_name_list):
    input_data = []
    golden_list = []
    for item in data_list:       
        input_data.append(item[0])
        tmp_label = [standard_name_list.index(i) for i in item[1]]
        golden_list.append(tmp_label + (7-len(tmp_label))*[-1])

    df = pd.DataFrame({'input':input_data, 'labels':golden_list})
    dataset = datasets.Dataset.from_pandas(df)
    encoded_dataset = dataset.map(preprocess_function,num_proc=8)
    return encoded_dataset

def get_recall_datasets():
    train_list = read_recall_data(args.data_path+"train.json")
    dev_list = read_recall_data(args.data_path+"dev.json")
    test_list = read_recall_data(args.data_path+"test.json")
    sdandard_list = list(normalized_words_set)
    data_list=dict()
    data_list["train"] = get_triplet_dataset(train_list,sdandard_list,args.neg_num)
    data_list["dev"] = get_pair_dataset(dev_list,sdandard_list)
    return data_list,sdandard_list,tokenizer