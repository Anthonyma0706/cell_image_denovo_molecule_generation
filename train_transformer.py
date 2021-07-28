import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter, OrderedDict
import argparse
import gc
from pytorch_lightning import callbacks

import scipy as sp
import pandas as pd
import numpy as np

import Levenshtein
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
from CFG import CFG

from data import TrainDataset, ValidDataset,Tokenizer # TestDataset
from transforms import get_train_transforms,get_val_transforms
from model import Encoder,DecoderWithAttention,DecoderWithTransformer
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2


tokenizer = torch.load(CFG.tokenizer_path)
from torchmetrics import Metric

class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("score", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, y_true, y_pred):
        scores = []
        for true, pred in zip(y_true, y_pred):
            score_1 = Levenshtein.distance(true, pred)
            scores.append(score_1)
        self.score = torch.tensor(np.mean(scores)).cuda()
        
    def compute(self):
        return self.score

class BMSPLModel(pl.LightningModule):
    def __init__(self, CFG, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG
        self.tokenizer = tokenizer
        self.accuracy = MyAccuracy()
        
        ### Encoder
        self.encoder = Encoder(CFG.encoder_name, pretrained=True)
                
        print(f'vocab_size is {len(tokenizer)}')
        ### Decoder
        if CFG.decoder_mode == 'lstm': ### decoder with LSTM attention
            self.decoder = DecoderWithAttention(
                attention_dim=CFG.attention_dim,
                embed_dim=CFG.embed_dim,
                decoder_dim=CFG.decoder_dim,
                vocab_size=len(tokenizer),
                dropout=CFG.dropout,
                encoder_dim=CFG.encoder_dim)
            
        elif CFG.decoder_mode == 'transformer': ### decoder with transformer
            self.decoder = DecoderWithTransformer(
                image_dim=CFG.image_dim,
                text_dim=CFG.text_dim,
                decoder_dim=CFG.decoder_dim,
                ff_dim=CFG.ff_dim,
                vocab_size=len(tokenizer),
                num_layer=CFG.num_layer,
                num_head=CFG.num_head,
                max_length=CFG.max_length,
                dropout=CFG.dropout)
        else:
            print(f'Decoder mode {CFG.decoder_mode} not supported, please select decoder mode in [lstm,transformer]')
    
            ### loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    
    def forward(self, x):
        features = self.encoder(x)
        predictions = self.decoder.predict(features, self.CFG.max_length, self.tokenizer)
        return predictions
        
    def get_scheduler(self,optimizer):
        if self.CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.CFG.factor, \
                    patience=self.CFG.patience, verbose=True, eps=self.CFG.eps)
        elif self.CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.CFG.T_max, eta_min=self.CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.CFG.T_0, T_mult=1, eta_min=self.CFG.min_lr, last_epoch=-1)
        return scheduler
    
    def configure_optimizers(self):
        encoder_optimizer = Adam(self.encoder.parameters(), lr=self.CFG.encoder_lr, weight_decay=self.CFG.weight_decay, amsgrad=False)
        encoder_scheduler = self.get_scheduler(encoder_optimizer)
        
        decoder_optimizer = Adam(self.decoder.parameters(), lr=self.CFG.decoder_lr, weight_decay=self.CFG.weight_decay, amsgrad=False)
        decoder_scheduler = self.get_scheduler(decoder_optimizer)
        return [encoder_optimizer, decoder_optimizer], [encoder_scheduler, decoder_scheduler]
    
    @property
    def automatic_optimization(self) -> bool:
        return False
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        images, labels, label_lengths = batch
        
        features = self.encoder(images)        
        if self.CFG.decoder_mode == 'lstm':
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(features, labels, label_lengths)
            targets = caps_sorted[:, 1:]
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = self.criterion(predictions, targets)
        elif self.CFG.decoder_mode == 'transformer':
            predictions, caps_sorted, decode_lengths = self.decoder(features, labels, label_lengths)
            targets = caps_sorted[:, 1:]
            decode_lengths = [l - 1 for l in decode_lengths]
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = self.criterion(predictions, targets)
        else:
            print('please select decoder mode in [lstm,transformer]')
        
        
        if self.CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        self.manual_backward(loss)
        
        opt_encoder, opt_decoder = self.optimizers()
        accumulate_gradient_batches = batch_idx % self.CFG.gradient_accumulation_steps == 0
        ### accumulate gradient batches
        if accumulate_gradient_batches:
            opt_encoder.step()
            opt_decoder.step()
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
    
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output
    
    def validation_step(self, batch, batch_idx):
        images, text_labels = batch
        
        features = self.encoder(images)
        # print(features.shape)
        predictions = self.decoder.predict(features, self.CFG.max_length, self.tokenizer)
        if CFG.decoder_mode == 'lstm':
            predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        elif CFG.decoder_mode == 'transformer':
            predicted_sequence = predictions.data.cpu().numpy()

        _text_preds = tokenizer.predict_to_inchi(predicted_sequence)
        
        return {'text_preds': _text_preds, 'text_labels': text_labels}


    # def get_score(self,y_true, y_pred):
    #     scores = []
    #     for true, pred in zip(y_true, y_pred):
    #         score = Levenshtein.distance(true, pred)
    #         scores.append(score)
    #     avg_score = np.mean(scores)
    #     return avg_score

    def validation_epoch_end(self, outputs):
        text_labels = np.concatenate([x['text_labels'] for x in outputs])
        text_preds = np.concatenate([x['text_preds'] for x in outputs])
#         text_preds = [f"InChI=1S/{text}" for text in text_preds]
        
        # scoring
        self.accuracy(text_labels,text_preds)
        score = self.accuracy.compute()
        print(f'Epoch: {self.current_epoch}, Score: {score}')
        print(f"labels: {text_labels[:5]}")
        print(f"preds {text_preds[:5]}")
        
        tqdm_dict = {'score': score}
        output = OrderedDict({
            'score': score,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        
        return output

# ====================================================
# Train loop
# ====================================================
def train_loop(CFG, folds,fold, tokenizer, model_path = ''):
    print(f'=============== fold: {fold} training =============')
    print(f'Training with {CFG.decoder_mode} decoder, params batch_size={CFG.batch_size*CFG.gpus}, encoder_lr={CFG.encoder_lr}, decoder_lr={CFG.decoder_lr}, epochs={CFG.epochs}')
    #trn_idx = folds[folds['fold'] != fold].index
    trn_idx = folds['fold'].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    #valid_folds = valid_folds[:1000]   ###快速测试模型时使用
    # valid_labels = valid_folds['InChI'].values

    train_dataset = TrainDataset(CFG, train_folds, tokenizer, transform=get_train_transforms(CFG))
    #valid_dataset = TestDataset(CFG, valid_folds, transform=get_val_transforms(CFG))
    valid_dataset = ValidDataset(CFG, valid_folds, transform=get_val_transforms(CFG))


    def bms_collate(batch):
        imgs, labels, label_lengths = [], [], []
        for data_point in batch:
            imgs.append(data_point[0])
            labels.append(data_point[1])
            label_lengths.append(data_point[2])
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
        return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
                          drop_last=True, shuffle=True, pin_memory=True, collate_fn=bms_collate)

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
                          drop_last=False, shuffle=False, pin_memory=True)
    
    model = BMSPLModel(CFG, tokenizer)
    
    # load weights!
    #model_path = './model/last_719_40pts.ckpt'
#model_path = './model/fold=0_epoch=09_score=43.3774.ckpt'
    if model_path != '':
        model = model.load_from_checkpoint(checkpoint_path=model_path, map_location=torch.device('cpu'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    # model = model.load_from_checkpoint(checkpoint_path="./model/fold=0_epoch=03_score=7.0143.ckpt")
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./model',
        filename='{fold}_{epoch:02d}_{score:.4f}',
        save_last = True,
        save_top_k= -1,  # save all models
        verbose=True,
        monitor='score',
        mode='min'
    )

#     logger = TensorBoardLogger(
#         save_dir = CFG.save_dir,
#         # version=1,
#         name='lightning_logs'
#     )

    # early_stop_callback = EarlyStopping(
    #     monitor='score',
    #     min_delta=100000.0,
    #     patience=3,
    #     verbose=False,
    #     mode='min',
    # )
    
    trainer = pl.Trainer(
        gpus=CFG.gpus,
        # gpus=[6,7],
        precision=CFG.precision,
        max_epochs=CFG.epochs,
        num_sanity_val_steps=1 if CFG.debug else 0,  ##开始训练前若干个batches的validation
        # checkpoint_callback=checkpoint_callback,
        #limit_train_batches=0.01,     ###快速测试模型时使用
        check_val_every_n_epoch = 1,  
        auto_select_gpus=True,
        accelerator='ddp', #for multi gpus
        sync_batchnorm=True,
        callbacks=[checkpoint_callback]
    
#         logger=logger,
        # callbacks=[early_stop_callback],
    )
    
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_loader)

if __name__ =="__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= '2,4'

    pl.seed_everything(seed=CFG.seed)

    folds = pd.read_csv(CFG.train_folds_path)
    
    #folds['folds'] 
    # print(folds)
    #if CFG.debug:
#         CFG.epochs = 1
    #    folds = folds.sample(n=6046142, random_state=CFG.seed).reset_index(drop=True)  ##6046142
   # print(folds.groupby(['fold']).size())
    
    #model_path = './model/last_719_40pts.ckpt'
    model_path = ''
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            train_loop(CFG, folds, fold, tokenizer, model_path)



