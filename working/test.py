#!/usr/bin/env python
# coding: utf-8

# In[2]:
from config import CFG
from importlib.metadata import distribution

# ====================================================
# Library
# ====================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import time

import joblib
import warnings
import glob
warnings.filterwarnings("ignore")
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
#pip install iterative-stratification==0.1.7
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler

from dataset import collate, TrainDataset
from utils import MCRMSE, get_score, get_logger, seed_everything, AverageMeter, timeSince, synchronize
from model import CustomModel

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # ====================================================
    # Data Loading
    # ====================================================
    test = pd.read_csv('../input/feedback-prize-english-language-learning/train_extra.csv')
    # test = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')

    return test

def set_tokenizer():
    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    # tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    return tokenizer

def init(CFG, local_rank, OUTPUT_DIR):

    test = load_data()

    tokenizer = set_tokenizer()

    # ====================================================
    # Define max_len
    # ====================================================
    lengths = []
    tk0 = tqdm(test['full_text'].fillna("").values, total=len(test))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    CFG.max_len = max(lengths) + 2 # cls & sep
    
    return test


def my_gather(output, local_rank, world_size):
    # output must be a tensor on the cuda device
    # output must have the same size in all workers
    result = None
    if local_rank == 0:
        result = [torch.empty_like(output) for _ in range(world_size)]
    torch.distributed.gather(output, gather_list=result, dst=0)
    return result

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


def valid_fn(valid_loader, model, device, local_rank, world_size, valid_dataset):

    model.eval()
    preds = []
    start = end = time.time()
    for step, inputs in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            y_preds = model(inputs)

        preds.append(y_preds.detach())


    # CPMP gather on worker 0
    predictions = torch.cat(preds) # CPMP
    if world_size == 1:
        return predictions.to('cpu').numpy()
    # CPMP gather on worker 0
    predictions = my_gather(predictions, local_rank, world_size)
    if local_rank == 0:
        predictions = torch.stack(predictions)
        # DistributedSampler interleaves workers
        # transpose restores the original order
        _, _, t = predictions.shape
        predictions = predictions.transpose(0, 1).reshape(-1, t) 
        # DistributedSampler pads the dataset to get a multiple of world size
        predictions = predictions[:len(valid_dataset)]
        return  predictions
    else:
        return  None

# ====================================================
# train loop
# ====================================================
def infer_fn(folds, fold, local_rank, distribution, world_size):
    # ====================================================
    # loader
    # ====================================================   
    valid_dataset = TestDataset(CFG, folds)

    val_sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                            num_replicas=world_size,
                                                            rank=local_rank,
                                                            shuffle=False,
                                                            seed=42,
                                                            drop_last=False,)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              sampler=val_sampler,
                              num_workers=CFG.num_workers, pin_memory=False, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=False)
    print('model loading...',OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
    state = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    
    model.load_state_dict(state['model'])
    model.to(device) 
    if distribution:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    find_unused_parameters=False)

    # ====================================================
    # loop
    # ====================================================

    predictions = valid_fn(valid_loader, model, device, local_rank, world_size, valid_dataset)
        
    return predictions

if __name__ == '__main__':
    OUTPUT_DIR = f'./exp/{CFG.EXP_NAME}/'  
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  


    parser = argparse.ArgumentParser(description="nn")

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    local_rank = args.local_rank
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    test = init(CFG, local_rank, OUTPUT_DIR)


    seed_everything(seed=42+local_rank)
    
    import pickle
    oof_df = pd.DataFrame()
    predictions = []
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            pred = infer_fn(test, fold, args.local_rank, distributed, num_gpus)
            # predictions.append(pred)
            with open(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_pred.pkl", 'wb') as  f:
                pickle.dump(pred, f)
            print(pred.shape)
            test[[f"pred_{c}_{fold}" for c in CFG.target_cols]] = pred
    # predictions = np.stack(predictions, axis=0).mean(axis=0)
    # predictions = torch.mean(predictions, dim=0).to('cpu').numpy()
    
    # labels = test[CFG.target_cols].values
    # score, scores = get_score(labels, predictions)
    # print(score, scores)
    # print(test.head(5))
    # test[CFG.target_cols] = predictions
    print(test.head(5))
    test.to_csv('../input/feedback-prize-english-language-learning/train_extra_withlabels.csv', index=False)








