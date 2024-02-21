import logging
import os
# import transformers
# import torch
# from torch.optim import AdamW

class Logger():
    def __init__(self,output_path,log_name):
        self.data_format = "%(asctime)s - %(levelname)s - %(message)s"
        self.output_path = output_path
        self.log_name = log_name

        self.init()

    def init(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        formatter = logging.Formatter(self.data_format)

        file_handler = logging.FileHandler(os.path.join(self.output_path,'{}.log'.format(self.log_name)))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        self.logger = logger

def print_args(args, logger):
    logger.info('#'*20 + 'Arguments' + '#'*20)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        logger.info('{}:{}'.format(k, v))

# def get_optimizer_and_scheduler(model, t_total, lr, warmup_steps, eps=1e-6, optimizer_class=AdamW, scheduler='WarmupLinear'):
#     def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
#         """
#         Returns the correct learning rate scheduler
#         """
#         scheduler = scheduler.lower()
#         if scheduler == 'constantlr':
#             return transformers.get_constant_schedule(optimizer)
#         elif scheduler == 'warmupconstant':
#             return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
#         elif scheduler == 'warmuplinear':
#             return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
#         elif scheduler == 'warmupcosine':
#             return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
#         elif scheduler == 'warmupcosinewithhardrestarts':
#             return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
#         else:
#             raise ValueError("Unknown scheduler {}".format(scheduler))

#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01, 'lr':lr},
#         {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0, 'lr':lr},
#     ]

#     local_rank = -1
#     if local_rank != -1:
#         t_total = t_total // torch.distributed.get_world_size()

#     optimizer_params = {'lr': lr, 'eps': eps}
#     optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
#     scheduler_obj = get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)
#     # scheduler_obj = None
#     return optimizer, scheduler_obj