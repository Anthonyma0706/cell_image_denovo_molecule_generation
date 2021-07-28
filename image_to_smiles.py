import os
from matplotlib import pyplot as plt

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('./pytorch-image-models-master')

import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import gc
import re
import math
import time
import random
import shutil
import pickle
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import Levenshtein
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from transforms import get_test_transforms
import timm

import warnings 
warnings.filterwarnings('ignore')

from data import TestDataset
from CFG import CFG
import cv2
import pytorch_lightning as pl
from model import Encoder,DecoderWithAttention,DecoderWithTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Rename PATHs ###
#img_root = "/gxr/fanyang/Molecular_image_benchmark/benchmark"
#submission_root = "/gxr/fanyang/projects/Efficient_transformer_transformer/benchmark_data/"

#image_list = os.listdir(img_root)
#pd.DataFrame(image_list,columns=["image_id"]).to_csv('{}benchmark_data_tianli.csv'.format(submission_root))


#test = pd.read_csv('./cil_data/df_info_meta_test.csv')
#test = pd.read_csv('./cil_data/df_info_meta_all_v2_sliced.csv') # check training accuracy


#test = pd.read_csv('./split_data/testset_by_cluster.csv')
#test['new_file_path'] = test['npy_path']

# for cDNA testset
test = pd.read_csv('./split_data/cDNA_testset_to_use.csv')

test['new_file_path'] = test['img_path']
test['npy_path'] = test['img_path']
test['smiles'] = 'ccc' * test.shape[0]

# adjust the testing size
#test = test.iloc[:2,:]
 

#def get_test_file_path(image_id):
#    return "{}/{}".format(
#        img_root,image_id 
#    )
#test['new_file_path'] = test['image_id'].apply(get_test_file_path)




print('begin Now:')

class Tokenizer(object):
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
    
    def predict_caption(self, sequence):
        caption = ''
        #print(sequence)
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            # print(type(i),i)
            caption += self.itos[i.item()]
        return caption
    
    def predict_captions(self, sequences):
        captions = []
       # print(sequences)
        #print(sequences.size())
        #sequences = torch.reshape(sequences,(1,-1))
        #print(sequences.size())
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

tokenizer = torch.load('./commercial_data/tokenizer_smiles_new.pth')
print(f"tokenizer.stoi: {tokenizer.itos}")

class BMSPLModel(pl.LightningModule):
    def __init__(self, CFG, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG
        self.tokenizer = tokenizer
        
        ### Encoder
        self.encoder = Encoder(CFG.encoder_name, pretrained=True)
        ### Decoder
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    
    def forward(self, x):
        features = self.encoder(x)
        predictions = self.decoder.predict(features, self.CFG.max_length, self.tokenizer)
        return predictions
        

def inference(test_loader, pl_model, tokenizer, device):
    pl_model.eval()
    
    # a list of list of list
    #text_preds_sampled_all = [] # a list of sample_size:10 of text_preds
    
    #text_preds = []
    
    tk0 = tqdm(test_loader, total=len(test_loader))
    
    num_images = len(test_loader)
    #################
    sample_size = 200
    text_preds_img_by_sample = np.zeros((num_images ,sample_size), dtype = '<U59')
    
    #################
    
    for j, images in enumerate(tqdm(tk0)):
        images = images.to(device)
        
        with torch.no_grad():
            predictions = pl_model(images)
            #print(f'prediction shape is : {predictions.shape}')
           # print(f'prediction is : {predictions}')
        
        text_preds_n_sample_per_image = []
        # Get the ith SAMPLE prediction of (ALL THE IMAGES in batch) -> Now is 1
        for i in range(sample_size): # predictions.shape[0] predict a smiles for each SAMPLE, size is 10 for now
            #_text_preds = tokenizer.predict_captions(predictions) 
            _text_preds = tokenizer.predict_captions(predictions[i])
            #print(f'_text_preds is {_text_preds}')
            
            text_preds_n_sample_per_image.append(_text_preds)  
            
        # We will get 10 of them at the end of LOOP 
        text_preds_n_sample_per_image = np.concatenate(text_preds_n_sample_per_image)  # 10 sample predictions for ONE IMAGE
        
        text_preds_n_sample_per_image.reshape(-1,sample_size)
        #print(f'the shape of text_preds_sample is (should be (1, sample_size)): {text_preds_n_sample_per_image.shape}')
        #print(f'text_preds_n_sample_per_image is : {text_preds_n_sample_per_image}')
        text_preds_img_by_sample[j] = text_preds_n_sample_per_image      
        
        
    #text_preds = np.concatenate(text_preds)
    #print(f'text_preds_img_by_sample SHAPE is {text_preds_img_by_sample.shape}')
    #print(f'text_preds_img_by_sample is {text_preds_img_by_sample}')
    #return text_preds
    
    return text_preds_img_by_sample


# To update the testset prediction on a specific channel (Hoechst, ERSyto, ERSyto Bleed): IMPORTANT TO FOLLOW
########################################################
########################################################
# 1. change the model weight path to the specific channel ('./model/last_725_1channel_ERsyto.ckpt')

# 2. Change data.py: change the image channel

# 3. change CGF.py: let batch_size = 1; let num_channels = 1 (image = image[1] # for ER syto)
########################################################
########################################################


# 4. (if just finished training): change predict() function in class DecoderWithTransformer() in model.py

########################################################
########################################################
# 5. change the saved file name below !!!
########################################################
########################################################

pl.seed_everything(seed=CFG.seed)
pl_model = BMSPLModel(CFG, tokenizer)

## load pl model weights
#model_path = './model/last_719_40pts.ckpt'
#model_path = './model/last_722_3channels.ckpt'
#model_path = './model/last_725_1channel_Hoechst.ckpt'

model_path = './model/last_727_1channel_ERsyto.ckpt'

pl_model = pl_model.load_from_checkpoint(checkpoint_path=model_path, map_location=torch.device('cpu'))
pl_model = pl_model.to(device)


test_dataset = TestDataset(CFG,test, transform=get_test_transforms(CFG))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
print('=========================before predictions========================')
predictions = inference(test_loader, pl_model, tokenizer, device)


# predictions is a numpy array if using Multinomial Sampling

predictions = predictions.reshape(-1)
df_pred = pd.DataFrame(predictions, columns = ['smiles_pred'])

img_path_list = list(test['img_path'])
new_path_l = []
sample_size = 200 ###### need to customize it
for path in img_path_list:
    for i in range(sample_size):
        new_path_l.append(path)

df_pred['img_path'] = new_path_l
   

#df_pred.to_csv('./split_data/MS_prediction_cDNA_testset_727.csv', index=False)

#df_pred.to_csv('./split_data/MS200_prediction_cDNA_testset_727_Hoechst.csv', index=False)
df_pred.to_csv('./split_data/MS200_prediction_cDNA_testset_727_ERSyto.csv', index=False)

print('========================= Prediction file saved ========================')




# For original testset: 

#test['smiles_pred'] = [f"{text}" for text in predictions]

#test[['npy_path', 'smiles_pred','smiles']].to_csv('./split_data/prediction_3channels_testset.csv', index=False)
#test[['img_path', 'smiles_pred']].to_csv('./split_data/prediction_cDNA_testset.csv', index=False)
#test[['img_path', 'smiles_pred', 'gene', 'plate_name', 'well_name']].to_csv('./split_data/prediction_cDNA_testset.csv', index=False)

#print(test[['npy_path', 'smiles']].head())
