from transformers import AutoTokenizer, AutoModel,PreTrainedModel, BertConfig
from torch import nn
from transformers.utils import ModelOutput
import torch
import faiss
from faiss import normalize_L2
import math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm


class TypeCLSModel(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_size = args.hidden_size
        self.max_lens = args.max_len
        self.label_num = args.label_num
        self.drop_rate = args.drop_rate
        self.encoder = AutoModel.from_pretrained(args.plm_name)
        config = AutoConfig.from_pretrained(args.plm_name)
        for param in self.encoder.parameters():
            param.requires_grad = True
        # self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size,padding_idx=0)
        # #其实max_len可以不用设置的太大，只要比最大句子的长度大一些就行了。
        # # self.position_embedding = PositionalEncoding(args.max_len, self.drop_rate, args.max_len)
        self.enhence_layer1 = nn.TransformerEncoderLayer(self.hidden_size, 6, self.hidden_size, self.drop_rate, "relu")
        self.enhence_layer2 = nn.TransformerEncoderLayer(self.hidden_size, 6, self.hidden_size, self.drop_rate, "relu")
        self.enhence_layer3 = nn.TransformerEncoderLayer(self.hidden_size, 6, self.hidden_size, self.drop_rate, "relu")

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*4, 1),
            # nn.Dropout(p=self.drop_rate),
            # nn.Linear(256, 1),
            # nn.Dropout(p=self.drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x1,x2,x3,x4):
        output1 = self.encoder(x1['input_ids'],attention_mask=x1['attention_mask'])[1]
        output2_e = self.encoder(x2['input_ids'],attention_mask=x2['attention_mask'],output_hidden_states=True).hidden_states[0]
        # print(output1)
        output3_e = self.encoder(x3['input_ids'],attention_mask=x3['attention_mask'],output_hidden_states=True).hidden_states[0]
        output4_e = self.encoder(x4['input_ids'],attention_mask=x4['attention_mask'],output_hidden_states=True).hidden_states[0]
        output2 = self.enhence_layer1(output2_e)[:,0]
        # print(output1)
        output3 = self.enhence_layer2(output3_e)[:,0]
        output4 = self.enhence_layer3(output4_e)[:,0]
        # print(output1.size())
        # print(pooler_output.size())
        ag = torch.cat((output1,output2,output3,output4),dim=1)
        # # print(pooler_output)
        logits = self.classifier(ag)
        return logits


@dataclass
class SiameseClassificationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None

class SiameseClassificationModel(PreTrainedModel):

    config_class = BertConfig
    supports_gradient_checkpointing = True

    def __init__(
            self, config, args, code
    ):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.bert = AutoModel.from_pretrained(self.config._name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        self.classification_loss_fun = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.config.hidden_size, 7)
        self.margin = 1.0
        self.code = code
        self.args = args
        self.code_embedding = None
        self.update_code_embedding()


    def _calculate_mean_pooled(self, last_hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_hidden_states = torch.sum(last_hidden_states * mask, -2)
        sum_mask = torch.clamp(mask.sum(-2), min=1e-12)
        mean_pooled = sum_hidden_states / sum_mask
        return mean_pooled

    def update_code_embedding(self):
        self.code_embedding = self.get_term_embedding(self.code)
        
    def get_term_embedding(self, term):
        if self.device == torch.device('cpu'):
            self.bert = self.bert.cuda()
        pred_batch = 5000
        embed_list = []
        for batch_num in tqdm(range(math.ceil(len(term)/pred_batch))):
            torch.cuda.empty_cache()
            start = batch_num*pred_batch
            end = (batch_num+1)*pred_batch
            batch_input = term[start:end]
            encode_input = self.tokenizer(batch_input,padding='max_length',max_length=48,return_tensors='pt',truncation=True)
            keys = list(encode_input)
            for k in keys:
                encode_input[k] = encode_input[k].to(self.bert.device)
            with torch.no_grad():
                batch_embed = self._calculate_mean_pooled(self.bert(**encode_input).last_hidden_state, encode_input['attention_mask'])
            embed_list.append(batch_embed)
        
        if self.device == torch.device('cpu'):
            self.bert = self.bert.to('cpu')

        return torch.cat(embed_list,dim=0).to('cpu')

    def get_distance(self,e1,e2):
        if self.args.distance == 'cosine':
            return 1 - F.cosine_similarity(e1,e2)
        elif self.args.distance == 'eu':
            return F.pairwise_distance(e1, e2, p=2)

    def faiss_distance(self, q_vectors, training_vectors, topN=50):
        
        d = self.config.hidden_size

        q_vectors = q_vectors.to('cpu')
        training_vectors = training_vectors.to(q_vectors.device)

        if self.args.distance == 'cosine':
            normalize_L2(training_vectors)
            normalize_L2(q_vectors)
            index=faiss.IndexFlatIP(d)        # the other index，需要以其他index作为基础
            index.train(training_vectors) 
            index.add(training_vectors)
            D, I =index.search(q_vectors, topN)
            score_list = D.tolist()
            index_list = I.tolist()
            return index_list,score_list
        elif self.args.distance == 'eu':
            index=faiss.IndexFlatL2(d)        # the other index，需要以其他index作为基础
            index.train(training_vectors) 
            index.add(training_vectors)
            D, I =index.search(q_vectors, topN)
            score_list = D.tolist()
            index_list = I.tolist()
            return index_list,score_list

    def triple_loss_fun(self, pos, neg):
        return F.relu(pos + self.margin - neg).mean()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        golden = None,
    ):
        if self.training:
            batch_size = input_ids.shape[0]
            cand_lengths = input_ids.shape[1] 

            input_ids = input_ids.reshape(batch_size*cand_lengths,-1)
            attention_mask = attention_mask.reshape(batch_size*cand_lengths,-1)
            token_type_ids = token_type_ids.reshape(batch_size*cand_lengths,-1)

            outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            )

            cls_output = outputs.pooler_output.reshape(batch_size, cand_lengths, -1)[:,0,:]
            cls_output = self.linear(cls_output)

            meaning_pooling_output = self._calculate_mean_pooled(outputs.last_hidden_state,attention_mask)

            meaning_pooling_output = meaning_pooling_output.reshape(batch_size, cand_lengths, -1)

            raw_output = meaning_pooling_output[:,0,:]
            pos_output = meaning_pooling_output[:,1,:]
            neg_output = meaning_pooling_output[:,2:,:]

            pos_distance = self.get_distance(raw_output, pos_output)
            neg_distance = self.get_distance(raw_output.unsqueeze(1).repeat(1, neg_output.shape[1], 1), neg_output)

            neg_distance = neg_distance.mean(1)

            triple_loss = self.triple_loss_fun(pos_distance, neg_distance)

            classification_loss = self.classification_loss_fun(cls_output, labels)

            loss = triple_loss + classification_loss
        
            return SiameseClassificationModelOutput(loss=loss, logits=cls_output, labels=labels)
        
        else:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            token_type_ids = token_type_ids.squeeze(1)

            outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            )

            cls_output = outputs.pooler_output
            cls_output = self.linear(cls_output)

            meaning_pooling_output = self._calculate_mean_pooled(outputs.last_hidden_state,attention_mask)

            index_list,score_list = self.faiss_distance(meaning_pooling_output, self.code_embedding.to(meaning_pooling_output.device))

            classification_loss = self.classification_loss_fun(cls_output, labels)

            return SiameseClassificationModelOutput(loss=classification_loss, logits=(cls_output, torch.tensor(index_list),torch.tensor(golden)), labels=labels)

