from core.utils import Logger
from argparse import ArgumentParser
from dataclasses import dataclass
import os
from transformers import TrainingArguments

def get_rr_config():
    return dict(
        OPENAI_API_BASE="https://fkopenai.9250.wtf/v1",
        API_KEY="sk-GZhfSiL8yOqS1Irq7fAe55E23e5e4c57B3Dd801d798bAbC6",
        FILTER_METHOD="m3e", # or bm25,None
        ADD_KNOWLEDGE=False,
    )

def get_rerank_config():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-data_name', default="match_cdn_", type=str)
    parser.add_argument('-data_path', default="data/recall_result/", type=str)
    parser.add_argument('-output_path', default=
                        "",
                        type=str)
    parser.add_argument('-model_path', default=
                        "",
                        type=str)
    parser.add_argument('-hidden_size', default=768, type=int)
    parser.add_argument('-max_len', default=100, type=int)
    parser.add_argument('-label_num', default=1, type=int)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-epoch_nums', default=10, type=int)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-version', default=24, type=int)
    parser.add_argument('-topk', default=100, type=int)
    parser.add_argument('-topk_predict', default=100, type=int)
    parser.add_argument('-drop_rate', default=0.2, type=float)
    parser.add_argument('-plm_name', default=
                        "bert-base-chinese", type=str)
    parser.add_argument('-threshold', default=0.5, type=float)
    parser.add_argument('-seed', default=34, type=float)
    args = parser.parse_known_args()[0]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    logger = Logger(args.output_path, 'main' + str(args.version)).logger
    # torch.manual_seed(args.seed)
    return args,logger


def get_recall_config():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-data_name', default="cdn", type=str)
    parser.add_argument('-data_path', default="data/CHIP-CDN-RR/", type=str)
    parser.add_argument('-corpus_path', default="data/knowledge_base/", type=str)
    parser.add_argument('-output_path', default=
                        "",
                        type=str)
    parser.add_argument('-model_path', default=
                        "",
                        type=str)
    parser.add_argument('-hidden_size', default=768, type=int)
    parser.add_argument('-max_len', default=64, type=int)
    parser.add_argument('-label_num', default=2, type=int)
    parser.add_argument('-batch_size', default=96, type=int)
    parser.add_argument('-epoch_nums', default=50, type=int)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-version', default=2, type=int)
    parser.add_argument('-topk', default=100, type=int)
    parser.add_argument('-topk_predict', default=100, type=int)
    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-margin', default=1.0, type=float)
    parser.add_argument('-neg_num', default=95, type=int)
    parser.add_argument('-neg_sample', default='online', type=str)
    parser.add_argument('-distance', default='eu', type=str)
    parser.add_argument('-metric_name', default='hit_rate', type=str)
    parser.add_argument('-plm_name', default=
                        "chinese_roberta_wwm_ext_L-12_H-768_A-12", type=str)
    parser.add_argument('-threshold', default=0.5, type=float)
    parser.add_argument('-seed', default=34, type=float)
    args = parser.parse_known_args()[0]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    training_args = TrainingArguments(
            args.output_path,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            learning_rate=args.learning_rate,
            do_eval = True,
            logging_steps = 10,
            eval_steps = 250,
            save_steps = 250,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=256,
            num_train_epochs=20,
            save_total_limit=2,
            seed = 42,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_name,
        )
    return args,training_args
    