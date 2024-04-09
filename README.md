# RRNorm

### RR.ipynb
* 用GPT拆分，中间结果为 `data/1_recognization_result`
* 规则识别和筛选，重组结果为 `data/2_restructure_result`
### recall.ipynb
* 召回模型训练，训练数据集为 `data/CHIP-CDN-RR`
* 召回预测，召回结果为 `data/3_recall_result`
### selection.ipynb
* 精排模型训练和预测，训练数据是召回结果

### 其他data
* `data/knowledge_baset`：标准库原始术语以及识别成分后的结果
* `data/origin_data`：原始CHIP-CDN数据集


### 其他code
* 一些评估和logger
