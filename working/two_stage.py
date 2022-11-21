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
import pickle
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
from utils import MCRMSE, get_score, get_logger, seed_everything, AverageMeter, timeSince, synchronize, FGM
from model import CustomModel

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # ====================================================
    # Data Loading
    # ====================================================
    train = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
    test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

    # ====================================================
    # CV split
    # ====================================================
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)

    return train, test, submission

def set_tokenizer(CFG):
    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    # tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    return tokenizer

def init(CFG, local_rank, dir):

    train, test, submission = load_data()

    if local_rank == 0:
        LOGGER = get_logger(filename=dir+'two_stage')
        print(train.groupby('fold').size())
        print(f"train.shape: {train.shape}")
        print(f"test.shape: {test.shape}")
        print(f"submission.shape: {submission.shape}")
    else:
        LOGGER = None

    return LOGGER, train, test, submission


# ====================================================
# Loss
# ====================================================
class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def my_gather(output, local_rank, world_size):
    # output must be a tensor on the cuda device
    # output must have the same size in all workers
    result = None
    if local_rank == 0:
        result = [torch.empty_like(output) for _ in range(world_size)]
    torch.distributed.gather(output, gather_list=result, dst=0)
    return result


def valid_fn(valid_loader, model, device, local_rank, world_size, valid_dataset):
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model.forward(inputs, extract_feature = True)
            # print(y_preds.shape)

        preds.append(y_preds.detach())
        end = time.time()

    # CPMP gather on worker 0
    predictions = torch.cat(preds) # CPMP
    if world_size == 1:
        return predictions
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
        return predictions
    else:
        return None


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, local_rank, distribution, world_size):
    if local_rank == 0:
        LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[(folds['fold'] == fold) & (folds['valid'] == True)].reset_index(drop=True)
    folds = folds[folds['fold'] == fold].reset_index(drop=True)

    # ====================================================
    # CV split again
    # ====================================================
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    print(folds.groupby('fold').size())
    labels = folds[CFG.target_cols].values

    for idx in range(4): 
        print(f'extracting feature {idx}')
        train_folds = folds[folds['fold'] != idx].reset_index(drop=True)
        valid_folds = folds[folds['fold'] == idx].reset_index(drop=True)
        print(train_folds.shape, valid_folds.shape)
        
        CFG.batch_size=16
        
        train_dataset = TrainDataset(CFG, train_folds)
        valid_dataset = TrainDataset(CFG, valid_folds)

        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=world_size,
                                                                rank=local_rank,
                                                                shuffle=False,
                                                                seed=42,
                                                                drop_last=False)
        val_sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                num_replicas=world_size,
                                                                rank=local_rank,
                                                                shuffle=False,
                                                                seed=42,
                                                                drop_last=False,)

        train_loader = DataLoader(train_dataset,
                                batch_size=CFG.batch_size,
                                sampler=train_sampler,
                                num_workers=CFG.num_workers, pin_memory=False, drop_last=False)
        valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.batch_size,
                                sampler=val_sampler,
                                num_workers=CFG.num_workers, pin_memory=False, drop_last=False)

        # ====================================================
        # model & optimizer
        # ====================================================
        all_train_pred = []
        all_valid_pred = []
        for model_name in tqdm(model_list, total=len(model_list)):
            CFG.model = model_name
            
            CFG.EXP_NAME = model_name.split('/')[-1]
            OUTPUT_DIR = f'./exp/{CFG.EXP_NAME}/'  
            if not os.path.exists(OUTPUT_DIR):
                raise ValueError(f'{OUTPUT_DIR}, assert output_dir exist')  

            
            tokenizer = set_tokenizer(CFG)
            # ====================================================
            # Define max_len
            # ====================================================
            if 'deberta-v3' in CFG.model:
                lengths = []
                tk0 = tqdm(train['full_text'].fillna("").values, total=len(train))
                for text in tk0:
                    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
                    lengths.append(length)
                CFG.max_len = max(lengths) + 2 # cls & sep
                if local_rank == 0:
                    LOGGER.info(f"max_len: {CFG.max_len}")
            else:
                CFG.max_len = 512

            model = CustomModel(CFG, config_path=None, pretrained=False)
            
            print('model loading...', OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            state = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                            map_location=torch.device('cpu'))
            
            model.load_state_dict(state['model'])
            model.to(device) 
            if distribution:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=False)

            start_time = time.time()

            # eval
            train_predictions = valid_fn(train_loader, model, device, local_rank, world_size, valid_dataset)
            valid_predictions = valid_fn(valid_loader, model, device, local_rank, world_size, valid_dataset)
            print(train_predictions.shape, valid_predictions.shape)
            all_train_pred.append(train_predictions.to('cpu').numpy())
            all_valid_pred.append(valid_predictions.to('cpu').numpy())
            # scoring
            if local_rank == 0:
                predictions = torch.cat([train_predictions, valid_predictions], dim=0)
                score, scores = get_score(labels, predictions.to('cpu').numpy())

                elapsed = time.time() - start_time

                LOGGER.info(f'elapsed: {elapsed}  Score: {score:.4f}  Scores: {scores}')

            torch.cuda.empty_cache()
        all_train_pred = np.concatenate(all_train_pred, axis=-1)
        all_valid_pred = np.concatenate(all_valid_pred, axis=-1)
        print(all_train_pred.shape, all_valid_pred.shape)

        with open(f'./fold_{fold}_{idx}_train_data.pkl', 'wb') as f:
            pickle.dump(all_train_pred, f)
        with open(f'./fold_{fold}_{idx}_valid_data.pkl', 'wb') as f:
            pickle.dump(all_valid_pred, f)    
     

def get_result(oof_df):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    if local_rank == 0:
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

if __name__ == '__main__':
    dir = f'./exp/'  
    if not os.path.exists(dir):
        raise ValueError(f'{dir}, assert output_dir exist') 

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
    
    LOGGER, train, test, submission = init(CFG, local_rank, dir=dir)
    if local_rank == 0:
        LOGGER.info("Using {} GPUs".format(num_gpus))
        LOGGER.info(args)

    seed_everything(seed=42+local_rank)
    
    model_list = ['roberta-base',
                  'roberta-large',
                  'bert-large-uncased',
                  'microsoft/deberta-base',
                  'microsoft/deberta-large',
                  'microsoft/deberta-v3-base',
                  'microsoft/deberta-v3-large',
                 ]
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(train, fold, args.local_rank, distributed, num_gpus)
        #         if local_rank == 0:
        #             oof_df = pd.concat([oof_df, _oof_df])
        #             LOGGER.info(f"========== fold: {fold} result ==========")
        #             get_result(_oof_df)
        # if local_rank == 0:
        #     oof_df = oof_df.reset_index(drop=True)
        #     LOGGER.info(f"========== CV ==========")
        #     get_result(oof_df)
        #     oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')





