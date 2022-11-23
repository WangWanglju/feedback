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
from utils import MCRMSE, get_score, get_logger, seed_everything, AverageMeter, timeSince, synchronize, FGM, AWP, revise_checkpoints
from model import CustomModel
from madgrad import MADGRAD
from torch.optim.swa_utils import (
        AveragedModel, update_bn, SWALR
    )
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # ====================================================
    # Data Loading
    # ====================================================
    train = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
    train['valid'] = True
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

def set_tokenizer():
    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    # tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    return tokenizer

def init(CFG, local_rank, OUTPUT_DIR):

    train, test, submission = load_data()

    if local_rank == 0:
        LOGGER = get_logger(filename=OUTPUT_DIR+'train')
        print(train.groupby('fold').size())
        print(f"train.shape: {train.shape}")
        print(f"test.shape: {test.shape}")
        print(f"submission.shape: {submission.shape}")
    else:
        LOGGER = None

    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)
        print(train.groupby('fold').size())
    
    # ====================================================
    # wandb
    # ====================================================
    if CFG.wandb and local_rank == 0:
        
        import wandb

        def class2dict(f):
            return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

        run = wandb.init(project='FB3-Public', 
                        name=CFG.model,
                        config=class2dict(CFG),
                        group=CFG.model,
                        job_type="train")


    tokenizer = set_tokenizer()


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


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, local_rank, world_size, swa_model=None, swa_scheduler=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    if CFG.awp == True:
        if not epoch < CFG.nth_awp_start_epoch:
            LOGGER.info(f'AWP training with epoch {epoch+1}')
        
        # Initializing the AWP class
        awp = AWP(
                model, 
                criterion, 
                optimizer,
                CFG.apex,
                adv_lr=CFG.awp_lr, 
                adv_eps=CFG.awp_eps
            )

    if CFG.fgm == True:
        fgm = FGM(model)

    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if CFG.fgm:
            # adversarial training
            fgm.attack() 
            with torch.cuda.amp.autocast(enabled = CFG.apex):
                y_preds = model(inputs)
                loss_adv = criterion(y_preds, labels)
                loss_adv.backward()
            fgm.restore()
        
        # AWP adversial attack for perturbation 
        if CFG.awp and CFG.nth_awp_start_epoch <= epoch:
            loss = awp.attack_backward(inputs, labels)
            scaler.scale(loss).backward()
            awp._restore()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                if not CFG.swa:
                    scheduler.step()
                else:
                    if (epoch+1) < CFG.swa_start:
                        scheduler.step()
                # scheduler.step()

        if CFG.swa and (epoch+1) >= CFG.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        # print(optimizer.param_groups[0]['lr'])

        end = time.time()
        if local_rank == 0:
            if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Time: {time} '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  {lr2:.8f}  '
                    'Momery: {memory:.2f}G'
                    .format(epoch+1, step, len(train_loader), 
                            time=time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()),
                            remain=timeSince(start, float(step+1)/len(train_loader)),
                            loss=losses,
                            grad_norm=grad_norm,
                            lr=optimizer.param_groups[0]['lr'],
                            lr2=optimizer.param_groups[-1]['lr'],
                            memory= torch.cuda.max_memory_allocated() / 1024.0**3))
            if CFG.wandb:
                wandb.log({f"[fold{fold}] loss": losses.val,
                        f"[fold{fold}] lr": scheduler.get_lr()[0]})
    if world_size == 1:
        return losses.avg
    loss_avg = torch.tensor([losses.avg], device=device)
    loss_avg = my_gather(loss_avg, local_rank, world_size)
    if local_rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
    else:
        loss_avg = None
    return loss_avg


def valid_fn(valid_loader, model, criterion, device, local_rank, world_size, valid_dataset):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.detach())
        end = time.time()
        if local_rank == 0:
            if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                    'Time: {time} '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'allocate momery: {memory:.2f}G'
                    .format(step, len(valid_loader),
                            time=time.strftime("%H:%M:%S", time.localtime()),
                            loss=losses,
                            remain=timeSince(start, float(step+1)/len(valid_loader)),
                            memory= torch.cuda.max_memory_allocated() / 1024.0**3))


    # CPMP gather on worker 0
    predictions = torch.cat(preds) # CPMP
    if world_size == 1:
        return losses.avg, predictions
    # CPMP gather on worker 0
    loss_avg = torch.tensor([losses.avg], device=device)
    loss_avg = my_gather(loss_avg, local_rank, world_size)
    predictions = my_gather(predictions, local_rank, world_size)
    if local_rank == 0:
        loss_avg = torch.cat(loss_avg).mean().item()
        predictions = torch.stack(predictions)
        # DistributedSampler interleaves workers
        # transpose restores the original order
        _, _, t = predictions.shape
        predictions = predictions.transpose(0, 1).reshape(-1, t) 
        # DistributedSampler pads the dataset to get a multiple of world size
        predictions = predictions[:len(valid_dataset)]
        return loss_avg, predictions
    else:
        return None, None
    return losses.avg, predictions

# ====================================================
# scheduler
# ====================================================
def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0, layerwise_learning_rate_decay=0.9):
    # optimizer_parameters = model.parameters()
    no_decay = ["bias", "LayerNorm.weight"]
    try:
        # optimizer_parameters = [
        #     {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         'lr': encoder_lr, 'weight_decay': weight_decay},
        #     {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
        #         'lr': encoder_lr, 'weight_decay': 0.0},
        #     {'params': [p for n, p in model.named_parameters() if "model" not in n],
        #         'lr': decoder_lr, 'weight_decay': 0.0}
        # ]

        optimizer_grouped_parameters = [
        {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            }
        ]
        # initialize lrs for every layer
        if 'deberta-v3' in CFG.model:
            layers = [model.model.embeddings] + list(model.model.encoder.layer) + [model.model.encoder.rel_embeddings] + [model.model.encoder.LayerNorm]
        elif 'roberta' in CFG.model:
            layers = [model.model.embeddings] + list(model.model.encoder.layer)
        elif 'deberta' in CFG.model:
            layers = [model.model.embeddings] + list(model.model.encoder.layer) + [model.model.encoder.rel_embeddings]
        else:
            layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()

        lr = encoder_lr

        for i, layer in enumerate(layers):
            print(f'layer {i} {[n for n, p in layer.named_parameters()][0]}: lr: {lr}')
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
                ]
            lr *= layerwise_learning_rate_decay
        return optimizer_grouped_parameters
    except Exception as e:
        print('distributed training...')
        optimizer_parameters = [
            {'params': [p for n, p in model.module.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.module.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.module.named_parameters() if "model" not in n],
                'lr': decoder_lr, 'weight_decay': 0.0}
        ]
    return optimizer_parameters

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, local_rank, distribution, world_size):
    if local_rank == 0:
        LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[(folds['fold'] == fold) & (folds['valid'] == True)].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                            num_replicas=world_size,
                                                            rank=local_rank,
                                                            shuffle=True,
                                                            seed=42,
                                                            drop_last=True)
    val_sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                            num_replicas=world_size,
                                                            rank=local_rank,
                                                            shuffle=False,
                                                            seed=42,
                                                            drop_last=False,)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              sampler=train_sampler,
                              num_workers=CFG.num_workers, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              sampler=val_sampler,
                              num_workers=CFG.num_workers, pin_memory=False, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    if local_rank == 0:
        print(model)
        LOGGER.info(model.config)
    model.to(device) 
    if distribution:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    find_unused_parameters=False)


    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay,
                                                layerwise_learning_rate_decay=CFG.llrd)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    # optimizer = MADGRAD(
    #         optimizer_parameters,
    #         lr=CFG.encoder_lr,
    #         eps=CFG.eps,
    #         weight_decay=CFG.weight_decay
    #     )
    
    # optimizer = PriorWD(optimizer, use_prior_wd=CFG.use_prior_wd)
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    if CFG.swa == True:
        # stochastic weight averaging
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                0.1 * averaged_model_parameter + 0.9 * model_parameter
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer, swa_lr=CFG.swa_learning_rate, 
            anneal_epochs=CFG.anneal_epochs, 
            anneal_strategy=CFG.anneal_strategy
        )
    else:
        swa_model = None
        swa_scheduler = None
    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    # criterion = nn.MSELoss(reduction="mean")
    
    best_score = np.inf

    for epoch in range(CFG.epochs):
        if distribution:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, local_rank, world_size, swa_model, swa_scheduler)
        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, local_rank, world_size, valid_dataset)

        
        # scoring
        if local_rank == 0:
            score, scores = get_score(valid_labels, predictions.to('cpu').numpy())

            elapsed = time.time() - start_time
            LOGGER.info(f'Time: {time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())}')
            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
            if CFG.wandb:
                wandb.log({f"[fold{fold}] epoch": epoch+1, 
                        f"[fold{fold}] avg_train_loss": avg_loss, 
                        f"[fold{fold}] avg_val_loss": avg_val_loss,
                        f"[fold{fold}] score": score})
            
            if best_score > score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                            OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
    if swa_model:
        update_bn(train_loader, swa_model, device=torch.device('cuda'))
        swa_avg_val_loss, swa_predictions = valid_fn(valid_loader, swa_model, criterion, device, local_rank, world_size, valid_dataset)
        score, scores = get_score(valid_labels, swa_predictions.to('cpu').numpy())

        LOGGER.info(f'SWA -  avg_val_loss: {swa_avg_val_loss:.4f}')
        LOGGER.info(f'SWA - Score: {score:.4f}  Scores: {scores}')
        torch.save({'model': swa_model.state_dict(),
            'predictions': swa_predictions},
            OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth")
        
        predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth", 
                                map_location=torch.device('cpu'))['predictions']
        valid_folds[[f"swa_pred_{c}" for c in CFG.target_cols]] = predictions

    if local_rank == 0:
        predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                                map_location=torch.device('cpu'))['predictions']
        valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

        torch.cuda.empty_cache()
        gc.collect()
        
        return valid_folds
    else:
        return None

def test_fn(folds, fold, local_rank, distribution, world_size):
    if local_rank == 0:
        LOGGER.info(f"========== fold: {fold} testing ==========")

    # ====================================================
    # loader
    # ====================================================
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

    valid_dataset = TrainDataset(CFG, valid_folds)

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

    if CFG.swa:
        print('model loading...', OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth")
        state = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth",
                    map_location=torch.device('cpu'))
        state['model'] = revise_checkpoints(state['model'])
    else:
        print('model loading...', OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
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
    # criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    criterion = nn.MSELoss(reduction="mean")
    
    best_score = np.inf

    start_time = time.time()
    
    # eval
    avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, local_rank, world_size, valid_dataset)

    
    # scoring
    if local_rank == 0:
        score, scores = get_score(valid_labels, predictions.to('cpu').numpy())

        elapsed = time.time() - start_time

        LOGGER.info(f'avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Score: {score:.4f}  Scores: {scores}')

    torch.cuda.empty_cache()
    gc.collect()
        
    return valid_folds


def get_result(oof_df, type = 'pred'):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"{type}_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    try:
        if local_rank == 0:
            LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
    except Exception as e:
        print(f'Score: {score:<.4f}  Scores: {scores}')

if __name__ == '__main__':
    OUTPUT_DIR = f'./exp/{CFG.EXP_NAME}/'  
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  

    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]

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
    
    LOGGER, train, test, submission = init(CFG, local_rank, OUTPUT_DIR)
    if local_rank == 0:
        LOGGER.info("Using {} GPUs".format(num_gpus))
        LOGGER.info(args)

    seed_everything(seed=42+local_rank)
    
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_sub = pd.read_csv('../input/feedback-prize-english-language-learning/fulldata_labels.csv')
                # train_sub['fold'] = 5
                train_sub[[f'{col}' for col in CFG.target_cols]] = train_sub[[f'pred_{col}_{fold}' for col in CFG.target_cols]]
                need_columns = ['text_id', 'full_text','fold'] + [f'{col}' for col in CFG.target_cols]
                train_sub = train_sub[need_columns]
                train_sub['valid'] = False
                # train_sub = train_sub.sample(frac=0.5)
                print('the number of extra data:', train_sub.shape[0])
                train_new = pd.concat([train, train_sub])

                _oof_df = train_loop(train_new, fold, args.local_rank, distributed, num_gpus)
                if local_rank == 0:
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(_oof_df, type='pred')
                    if CFG.swa:
                        LOGGER.info(f"SWA result ==========")
                        get_result(_oof_df, type='swa_pred')
        if local_rank == 0:
            oof_df = oof_df.reset_index(drop=True)
            LOGGER.info(f"========== CV ==========")
            get_result(oof_df)
            oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
            if CFG.swa:
                get_result(oof_df, type='swa_pred')
                oof_df.to_pickle(OUTPUT_DIR+'swa_oof_df.pkl')
    else:
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = test_fn(train, fold, args.local_rank, distributed, num_gpus)
        
    if CFG.wandb and local_rank == 0:
        wandb.finish()





