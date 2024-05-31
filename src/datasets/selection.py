from torch.utils.data import Dataset, DataLoader
import json
import re
from configure import get_rerank_config
args,_ = get_rerank_config()
corpus = {}
with open("data/knowledge_base/code.jsonl", "r", encoding='utf-8') as f:
    line = f.readline()
    while line:
        tmp = json.loads(line.strip())
        # corpus.append(.split('\t')[1])
        try:
            corpus[tmp["text"]] = json.loads(tmp["gpt_sub"])
        except:
            corpus[tmp["text"]] = [{"原词": tmp["text"], "类别": "发病内容"}]
        line = f.readline()
tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
class Dataset(Dataset):
    def __init__(self, X1,X2,X3,X4, y=None, flag='train',
                tokenizer=None):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.y = y
        self.flag = flag
        self.tokenizer = tokenizer
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.max_len = args.max_len

    def __len__(self):
        return len(self.X1)
    
    def my_mask(self,token_ids):
        mask = []
        for ids in token_ids:
            if ids==103 or ids == 0:
                mask.append(0)
            else:
                mask.append(1)
        return torch.tensor(mask)

    def __getitem__(self, index):
        # if self.flag=='train':
        # print(self.X[index],self.y[index])
        X1 = self.X1[index]
        X2 = self.X2[index]
        X3 = self.X3[index]
        X4 = self.X4[index]
        label = self.y[index]
        inputs1 = self.tokenizer(X1, padding='max_length',
                                 max_length=self.max_len, truncation=True,
                                 return_tensors='pt')
        inputs2 = self.tokenizer(X2, padding='max_length',
                                 max_length=self.max_len, truncation=True,
                                 return_tensors='pt')
        inputs3 = self.tokenizer(X3, padding='max_length',
                                 max_length=self.max_len, truncation=True,
                                 return_tensors='pt')
        inputs4 = self.tokenizer(X4, padding='max_length',
                                 max_length=self.max_len, truncation=True,
                                 return_tensors='pt')
        output_X1 = {}
        for i in inputs1:
            output_X1[i] = inputs1[i].squeeze()
        output_X2 = {}
        for i in inputs1:
            output_X2[i] = inputs2[i].squeeze()
        output_X2["attention_mask"] = self.my_mask(output_X2["input_ids"])
        output_X3 = {}
        for i in inputs1:
            output_X3[i] = inputs3[i].squeeze()
        output_X3["attention_mask"] = self.my_mask(output_X3["input_ids"])
        output_X4 = {}
        for i in inputs1:
            output_X4[i] = inputs4[i].squeeze()
        output_X4["attention_mask"] = self.my_mask(output_X4["input_ids"])
        return output_X1,output_X2,output_X3,output_X4, label

def construct_mask(text,targets):
    text_1 = text
    for target in targets:
        if target in text_1:
            start = text_1.find(target)
            end = start+len(target)
            text_1 = list(text_1)
            for p in range(start,end):
                # print(text[2])
                
                text_1[p] = "@"
            text_1 = "".join(text_1)
    # text = text.replace("@","[MASK]")
    text = list(text)
    for index,(t1,t) in enumerate(zip(text_1,text)):
        if t1!="@":
            text[index] = "@"
    text = "".join(text)
    text = text.replace("@","[MASK]")
    return text
# construct_mask("左膝退变伴游离体",["退变","游离体"])

def get_typecls_datasets(tokenizer):
    todo_list = ["dev","train","test"]
    data_list = {}
        
    for todo_set in todo_list:
        X1 = []
        X2 = []
        X3 = []
        X4 = []
        y = []
        recall_list = json.load(open("/mnt/data/smart_health_02/zhuyansha/data/CHIP-CDN-SR/recall_split_top20/" + todo_set + ".json"))
        split_list = json.load(open("/mnt/data/smart_health_02/zhuyansha/data/CHIP-CDN-SR/CHIP-CDN_" + todo_set + ".json"))
        label_list = json.load(open("/mnt/data/smart_health_02/zhuyansha/code/seg+gen/input/CHIP-CDN-rule_stage_0/CHIP-CDN_" + todo_set + ".json"))
        recall_dict = {}
        split_dict = {}
        for item in label_list:
            split_dict[item["text"]] = item["label_result"]
        for item in split_list:
            for sr in item["SR"]:
                sr = re.sub('\[.*?\]','',sr)
                tmp = []
                for t in split_dict[item["text"]]:
                    if t["原词"] in sr:
                        tmp.append(t)
                split_dict[sr] = tmp
        for item in recall_list:
            for candidate in item["candidates"]:
                split_label_m = split_dict[item["text"]]
                split_label_c = corpus[candidate]
                x2 = ""
                x3 = ""
                x4 = ""
                DC = []
                DS = []
                OC = []
                for tm in split_label_m:
                    if tm["类别"] == "发病内容":
                        DC.append(tm["原词"])
                    if tm["类别"] == "发病范围":
                        DS.append(tm["原词"])
                    if tm["类别"] == "操作内容":
                        OC.append(tm["原词"])
                x2+=construct_mask(item["text"],DC)+"[SEP]"
                x3+=construct_mask(item["text"],DS)+"[SEP]"
                x4+=construct_mask(item["text"],OC)+"[SEP]"
                
                DC = []
                DS = []
                OC = [] 
                for tc in split_label_c:
                    if tc["类别"] == "发病内容":
                        DC.append(tc["原词"])
                    if tc["类别"] == "发病范围":
                        DS.append(tc["原词"])
                    if tc["类别"] == "操作内容":
                        OC.append(tc["原词"])
                x2+=construct_mask(candidate,DC)
                x3+=construct_mask(candidate,DS)
                x4+=construct_mask(candidate,OC)
                
                X1.append(item["text"] + "[SEP]" +  candidate)
                X2.append(x2)
                X3.append(x3)
                X4.append(x4)
                
                if todo_set == "train":
                    if candidate in item["normalized_result"]:
                        y.append(1)
                    else:
                        y.append(0)
                elif todo_set == "dev":
                    y.append(item["text"]   +"###"+ candidate)
                else:
                    y.append(item["text"]  +"###"+ candidate)
            recall_dict[item["text"]] = item["candidates"]
        for item in split_list:
            candidates = []
            for sr in item["SR"]:
                sr = re.sub('\[.*?\]','',sr)
                candidates.extend(recall_dict[sr])
            for candidate in candidates:
                split_label_c = corpus[candidate]
                split_label_m = split_dict[item["text"]]
                x2 = ""
                x3 = ""
                x4 = ""
                DC = []
                DS = []
                OC = []
                for tm in split_label_m:
                    if tm["类别"] == "发病内容":
                        DC.append(tm["原词"])
                    if tm["类别"] == "发病范围":
                        DS.append(tm["原词"])
                    if tm["类别"] == "操作内容":
                        OC.append(tm["原词"])
                x2+=construct_mask(item["text"],DC)+"[SEP]"
                x3+=construct_mask(item["text"],DS)+"[SEP]"
                x4+=construct_mask(item["text"],OC)+"[SEP]"
                
                DC = []
                DS = []
                OC = [] 
                for tc in split_label_c:
                    if tc["类别"] == "发病内容":
                        DC.append(tc["原词"])
                    if tc["类别"] == "发病范围":
                        DS.append(tc["原词"])
                    if tc["类别"] == "操作内容":
                        OC.append(tc["原词"])
                x2+=construct_mask(candidate,DC)
                x3+=construct_mask(candidate,DS)
                x4+=construct_mask(candidate,OC)
                            
                X1.append(item["text"] + "[SEP]" +  candidate)
                X2.append(x2)
                X3.append(x3)
                X4.append(x4)
                if todo_set == "train":
                    if candidate in item["normalized_result"].split("##"):
                        y.append(1)
                    else:
                        y.append(0)
                elif todo_set == "dev":
                    y.append(item["text"]   +"###"+ candidate)
                else:
                    y.append(item["text"]  +"###"+ candidate)
        data_list[todo_set] = Dataset(X1,X2,X3,X4, y, flag='train', tokenizer=tokenizer)
    return data_list