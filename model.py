import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from CFG import CFG
import sys
# sys.path.append('../input/pytorchimagemodels210218/pytorch-image-models-master')
sys.path.append('./pytorch-image-models-master')
import timm
from tqdm import tqdm
from timm.models.tnt import tnt_s_patch16_224 
image_dim = 384
num_pixel=7*7
####### CNN ENCODER


class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.model_name = model_name
        if model_name == 'resnet18' or model_name == 'resnet34' or model_name == 'resnet200d':
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            n_ch = 5 #1
            self.cnn.conv1.weight = nn.Parameter(self.cnn.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.n_features = self.cnn.fc.in_features
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        if model_name == 'tf_efficientnet_b0_ns' or model_name == 'tf_efficientnet_b3_ns' or model_name == 'tf_efficientnet_b4_ns':
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.classifier.in_features
            self.cnn.global_pool = nn.Identity()
            self.cnn.classifier = nn.Identity()
            self.project = nn.Sequential(
                nn.Conv2d(1280,image_dim, kernel_size=1, bias=None),
                nn.BatchNorm2d(image_dim)
                # Swish()
                )
        
        if model_name == 'vit_base_patch16_224' or model_name == 'vit_base_patch16_384':
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            
            self.n_features = self.cnn.head.in_features 
#             self.cnn.norm = nn.Identity()
            self.cnn.head = nn.Identity()
            
        if model_name == 'tnt_s_patch16_224':
            self.cnn = tnt_s_patch16_224(pretrained=pretrained)
            n_ch = 1 #1 #3 #CFG.image_channels #5 #1
            self.cnn.pixel_embed.proj.weight = nn.Parameter(self.cnn.pixel_embed.proj.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            
        # if model_name == 'tnt_s_patch16_384':  # need modify tnt.py by yourself
        #     self.cnn = tnt_s_patch16_384(pretrained=False)
        
    def forward(self, x):
        if self.model_name == 'vit_base_patch16_224':
            B = x.shape[0]
#             x = 2 * x - 1 
            #print(f'input x size is {x.size()}')
        
            x = self.cnn.patch_embed(x)
            cls_tokens = self.cnn.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.cnn.pos_embed
            x = self.cnn.pos_drop(x)

            for blk in self.cnn.blocks:
                x = blk(x)

#             enc_self_attns = self.cnn.norm(x)[:, 0]
#             encoder_out = self.cnn.norm(x)[:, 1:]
            x = self.cnn.norm(x) ### {batch_size, 197, 768]
            
            return x
        
        elif self.model_name == 'tnt_s_patch16_224' or self.model_name == 'tnt_s_patch16_384':
            batch_size, C, H, W = x.shape
            x = 2 * x - 1  # ; print('input ',   x.size())

            pixel_embed = self.cnn.pixel_embed(x, self.cnn.pixel_pos)

            patch_embed = self.cnn.norm2_proj(self.cnn.proj(self.cnn.norm1_proj(pixel_embed.reshape(batch_size, self.cnn.num_patches, -1))))
            patch_embed = torch.cat((self.cnn.cls_token.expand(batch_size, -1, -1), patch_embed), dim=1)
            patch_embed = patch_embed + self.cnn.patch_pos
            patch_embed = self.cnn.pos_drop(patch_embed)

            for blk in self.cnn.blocks:
                pixel_embed, patch_embed = blk(pixel_embed, patch_embed) # 用pixel_embed一起不停迭代，最后得到好的patch_embed
            
            # normalize it in the end
            patch_embed = self.cnn.norm(patch_embed) #torch.Size([7, 197, 384])

            x = patch_embed
            
            return x
        
        else: # default: resnet 18
            bs = x.size(0)
            features = self.cnn(x)
            features = self.project(features)
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(bs, num_pixel, image_dim)
            
            return features

####### RNN DECODER

# attention module
class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

### LSTM decoder with attention
class DecoderWithAttention(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
 
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        :param caption_lengths: length of transformed sequence
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # embedding transformed sequence for vector
#         embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions).to(encoder_out.dtype) # for 16bit precision added by chen
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            attention_weighted_encoding = attention_weighted_encoding.to(encoder_out.dtype) ## added by chen for 16bit precision
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def predict(self, encoder_out, decode_lengths, tokenizer):
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(device) * tokenizer.stoi["<sos>"]
        embeddings = self.embedding(start_tockens)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(device)
        
#         eos = tokenizer.stoi['<eos>']
#         pad = tokenizer.stoi['<pad>']
        # predict sequence
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == tokenizer.stoi['<eos>']:  
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions
    
    def forward_step(self, prev_tokens, hidden, encoder_out, function):
        assert len(hidden) == 2
        h, c = hidden
        h, c = h.squeeze(0), c.squeeze(0)

        embeddings = self.embedding(prev_tokens)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = self.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1),
            (h, c))  # (batch_size_t, decoder_dim)
        preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

        hidden = (h.unsqueeze(0), c.unsqueeze(0))
        predicted_softmax = function(preds, dim=1)
        return predicted_softmax, hidden, None   


### Decoder with Transformer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
# Table 1: Post-LN Transformer v.s. Pre-LN Transformer
# ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE - ICLR 2020
# https://openreview.net/pdf?id=B1x8anVFPr

# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://scale.com/blog/pytorch-improvements
# Making Pytorch Transformer Twice as Fast on Sequence Generation.
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):   ###T代表smiles的长度
        batch_size, T, dim = x.shape
        #print(x.shape)
        #print(T)
        
        x = x + self.pos[:,:T]
        #print(x.shape)
        
        return x

#https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionEncode2D(nn.Module):
    def __init__(self, dim, width, height):
        super().__init__()
        assert (dim % 4 == 0)
        self.width  = width
        self.height = height

        dim = dim//2
        d = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        position_w = torch.arange(0., width ).unsqueeze(1)
        position_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim*2, height, width)

        pos[0,      0:dim:2, :, :] = torch.sin(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,      1:dim:2, :, :] = torch.cos(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,dim + 0:   :2, :, :] = torch.sin(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        pos[0,dim + 1:   :2, :, :] = torch.cos(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x + self.pos[:,:,:H,:W]
        return x
    

# def triangle_mask(size):
#     mask = 1- np.triu(np.ones((1, size, size)),k=1).astype('uint8')
#     mask = torch.autograd.Variable(torch.from_numpy(mask))
#     return mask

'''
triangle_mask(10)

mask
array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], dtype=uint8)
'''

# ------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

#layer normalization
class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps
    def forward(self, x):
        #return x
        z = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        x = self.alpha*z + self.bias
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.d_k = dim // num_head
        self.num_head = num_head
        self.dropout = dropout

        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def attention(self, q, k, v, mask):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # torch.Size([8, 4, 10, 10]) = batch_size, num_head, LqxLk
        if mask is not None:
            mask = mask.unsqueeze(1)
            #print(score.min())
            score = score.masked_fill(mask == 0, -6e4) #-65504
#             score = score.masked_fill(mask == 0, -half('inf'))
            # https://github.com/NVIDIA/apex/issues/93
            # How to use fp16 training with masked operations

        score = F.softmax(score, dim=-1)

        if self.dropout > 0:
            score = F.dropout(score, self.dropout, training=self.training)

        value = torch.matmul(score, v)
        return value


    def forward(self, q, k, v, mask=None):
        batch_size, T, dim = q.shape

        # perform linear operation and split into h heads
        k = self.k(k).reshape(batch_size, -1, self.num_head, self.d_k)
        q = self.q(q).reshape(batch_size, -1, self.num_head, self.d_k)
        v = self.v(v).reshape(batch_size, -1, self.num_head, self.d_k)

        # transpose to get dimensions batch_size * num_head * T * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        value = self.attention(q, k, v, mask)

        # concatenate heads and put through final linear layer
        value = value.transpose(1, 2).contiguous().reshape(batch_size, -1, self.dim)
        value = self.out(value)
        return value


#---
class TransformerEncodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

        self.attn = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.ff   = FeedForward(dim, ff_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x1 = self.norm1(x)
        x1 = self.attn(x1, x1, x1, x_mask) #self-attention
        x   = x + self.dropout1(x1)

        x2 = self.norm2(x)
        x2 = self.ff(x2)
        x  = x + self.dropout2(x2)
        return x

class TransformerEncode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.num_layer = num_layer

        self.layer = nn.ModuleList([
            TransformerEncodeLayer(dim, ff_dim, num_head) for i in range(num_layer)
        ])
        self.norm = Norm(dim)

    def forward(self, x, x_mask):
        for i in range(self.num_layer):
            x = self.layer[i](x, x_mask)
        x = self.norm(x)
        return x

#---
class TransformerDecodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.norm3 = Norm(dim)

        self.attn1 = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.attn2 = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.ff = FeedForward(dim, ff_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mem, x_mask, mem_mask):
        x1 = self.norm1(x)
        x1 = self.attn1(x1, x1, x1, x_mask)  # self-attention
        x  = x + self.dropout1(x1)

        if mem is not None:
            x2 = self.norm2(x)
            x2 = self.attn2(x2, mem, mem, mem_mask)  # encoder input
            x  = x + self.dropout2(x2)

        x3 = self.norm3(x)
        x3 = self.ff(x3)
        x  = x + self.dropout3(x3)
        return x

    def forward_last(self, x_last, x_cache, mem, mem_mask):

        x_last_norm = self.norm1(x_last)
        x1 = torch.cat([x_cache, x_last_norm], 1)
        x_cache = x1.clone() # update

        x1 = self.attn1(x_last_norm, x1, x1)
        x_last  = x_last + x1

        if mem is not None:
            x2 = self.norm2(x_last)
            x2 = self.attn2(x2, mem, mem, mem_mask)
            x_last = x_last + x2


        x3 = self.norm3(x_last)
        x3 = self.ff(x3)
        x_last = x_last + x3

        return x_last, x_cache


# https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
class TransformerDecode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.num_layer = num_layer

        self.layer = nn.ModuleList([
            TransformerDecodeLayer(dim, ff_dim, num_head) for i in range(num_layer)
        ])
        self.norm = Norm(dim)

    def forward(self, x, mem, x_mask=None, mem_mask=None):

        for i in range(self.num_layer):
            x = self.layer[i](x, mem, x_mask, mem_mask)

        x = self.norm(x)
        return x

    def forward_last(self, x_last, x_cache, mem,  mem_mask=None):
        batch_size,t,dim = x_last.shape
        assert(t==1)

        for i in range(self.num_layer):
            x_last, x_cache[i] = self.layer[i].forward_last(x_last, x_cache[i], mem, mem_mask)

        x_last = self.norm(x_last)
        return x_last, x_cache

    
### Decoder with pre transformer    
class DecoderWithTransformer(nn.Module):
    def __init__(self,image_dim,text_dim,decoder_dim,ff_dim,vocab_size,num_layer=3,num_head=8,max_length=300,dropout=0.5):
        
        """
        :param image_dim: input size of image network 
        :param text_dim: input size of text network 
        :param decoder_dim: input size of decoder network 
        :param ff_dim: input size of forwardfeed network 
        :param vocab_size: total number of characters used in training
        :param num_layer: decoder layer numbers
        :param num_head : head numbers 
        :param max_length : text max lenght 
        :param dropout: dropout rate
        """
        super(DecoderWithTransformer, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.decoder_dim = decoder_dim
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.max_length = max_length
        self.dropout = dropout
        
        
        self.image_encode = nn.Identity()
        
        self.text_pos   = PositionEncode1D(text_dim,max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        #---
        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)
    
    @torch.jit.unused
    def forward(self, image_embed, token, length):
        device = image_embed.device
        batch_size = image_embed.size(0)
        
#         image_embed = image_embed[:,1:]
#         enc_self_attns = image_embed[:,0]
        #---
        image_embed = self.image_encode(image_embed)

        length, sort_ind = length.squeeze(1).sort(dim=0, descending=True)
        decode_lengths = (length).tolist()
        
        image_embed = image_embed[sort_ind]
        token = token[sort_ind]
        max_length = max(decode_lengths)
        
        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed)

        text_mask = 1 - np.triu(np.ones((batch_size, max_length, max_length)), k=1).astype(np.uint8)

        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)).to(device)
        text_image_mask = None
        
        #----
        # <todo> mask based on length of token?
        # <todo> perturb mask as augmentation https://arxiv.org/pdf/2004.13342.pdf

        x = self.text_decode(text_embed, image_embed, text_mask, text_image_mask)
        logit = self.logit(x)
        
        return logit, token, length
    

    @torch.jit.export
    def predict_valid(self, image_embed, max_length, tokenizer):
        ####
        # predict with argmax
        ####
        image_dim = self.image_dim
        text_dim = self.text_dim
        decoder_dim = self.decoder_dim
        num_layer = self.num_layer
        num_head = self.num_head
        ff_dim = self.ff_dim
        vocab_size = self.vocab_size

        #---------------------------------
        device = image_embed.device
        batch_size = image_embed.size(0)
        
#         image_embed = image_embed[:,1:]
#         enc_self_attns = image_embed[:,0]

        image_embed = self.image_encode(image_embed)

        token = torch.full((batch_size, max_length), tokenizer.stoi['<pad>'],dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:,0] = tokenizer.stoi['<sos>']

        #-------------------------------------
        eos = tokenizer.stoi['<eos>']
        pad = tokenizer.stoi['<pad>']
        # https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py

        ## fast version
        cache = [torch.empty((batch_size,0,decoder_dim), device=device) for i in range(num_layer)]
        for t in range(max_length-1):
            #last_token = token [:,:(t+1)]
            #text_embed = self.token_embed(last_token)
            #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

            last_token = token[:, t]
            text_embed = self.token_embed(last_token)
            text_embed = text_embed + text_pos[:,t] #
            text_embed = text_embed.reshape(batch_size,1,text_dim)

            x, cache = self.text_decode.forward_last(text_embed[:, -1:], cache, image_embed)
            x = x.reshape(batch_size,decoder_dim)

            l = self.logit(x)
            k = torch.argmax(l, -1)  # predict max
            token[:, t+1] = k

            if ((k == eos) | (k == pad)).all():  break

        predictions = token[:, 1:]
        return predictions  
    
    
    def predict(self, image_embed, max_length, tokenizer):
        ####
        # predict with Multinomial Sampling
        #### 
        
        
        # SAMPLE SIZE
        #sample_size = 10
        sample_size = 200
        
        image_dim = self.image_dim
        text_dim = self.text_dim
        decoder_dim = self.decoder_dim
        num_layer = self.num_layer
        num_head = self.num_head
        ff_dim = self.ff_dim
        vocab_size = self.vocab_size

        #---------------------------------
        device = image_embed.device
        batch_size = image_embed.size(0)

        image_embed = self.image_encode(image_embed)
        # note that this must be max_length -1 since we will slice off the first sos vocab
        
        predictions_whole = torch.full((sample_size, batch_size, max_length - 1), tokenizer.stoi['<pad>'],dtype=torch.long, device=device)
        print(f'=============== Multinomial Sampling ... with sample size {sample_size}================')
                                        
        for i in tqdm(range(sample_size)):
            token = torch.full((batch_size, max_length), tokenizer.stoi['<pad>'],dtype=torch.long, device=device)
            text_pos = self.text_pos.pos
            token[:,0] = tokenizer.stoi['<sos>']

            #-------------------------------------
            eos = tokenizer.stoi['<eos>']
            pad = tokenizer.stoi['<pad>']
            # https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py

            ## fast version
            cache = [torch.empty((batch_size,0,decoder_dim), device=device) for i in range(num_layer)]
            for t in range(max_length-1):
                last_token = token[:, t]
                text_embed = self.token_embed(last_token)
                text_embed = text_embed + text_pos[:,t] #
                text_embed = text_embed.reshape(batch_size,1,text_dim)

                x, cache = self.text_decode.forward_last(text_embed[:, -1:], cache, image_embed)
                x = x.reshape(batch_size,decoder_dim)

                l = self.logit(x) # the probability tensor of each vocab
                probabilities = torch.softmax(l, dim=-1)
                ####################################
                # now we do not want the MAX -> Greedy search
                #k_max = torch.argmax(l, -1)  # predict max
                #print(f'the max is {k_max}')
                                        
                # now we want to use Multinomial
                k = torch.multinomial(probabilities, num_samples=1)
                #print(f'the sampled is {k} ')
                ####################################
                token[:, t+1] = k
                #print(f'the token size is {token.shape}')
               # print(f'the token now is {token}')
                if ((k == eos) | (k == pad)).all():  break

            predictions = token[:, 1:] # ignore the start token
            predictions_whole[i,:,:] = predictions
        
        #print(f'the token is {token[:, :]}')
        
       # print(predictions_whole[0] == predictions_whole[1])
        return predictions_whole #predictions     
    
    
    
    
    # slow version
#         for t in range(max_length-1):
#             last_token = token [:,:(t+1)]
#             text_embed = self.token_embed(last_token)
#             text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous() #text_embed + text_pos[:,:(t+1)] #
        
#             text_mask = np.triu(np.ones((t+1, t+1)), k=1).astype(np.uint8)
#             text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)
        
#             x = self.text_decode(text_embed, image_embed, text_mask)
#             x = x.permute(1,0,2).contiguous()
        
#             l = self.logit(x[:,-1])
#             k = torch.argmax(l, -1)  # predict max
#             token[:, t+1] = k
#             if ((k == eos) | (k == pad)).all():  break

    '''
    
    @torch.jit.export
    def predict_K(self, image_embed, max_length, tokenizer,top_k=5):
        ####
        # predict with argmax
        ####
        image_dim = self.image_dim
        text_dim = self.text_dim
        decoder_dim = self.decoder_dim
        num_layer = self.num_layer
        num_head = self.num_head
        ff_dim = self.ff_dim
        vocab_size = self.vocab_size

        #---------------------------------
        device = image_embed.device
        batch_size = image_embed.size(0)

        image_embed = self.image_encode(image_embed)

        token = torch.full((batch_size, max_length), tokenizer.stoi['<pad>'],dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:,0] = tokenizer.stoi['<sos>']

        #-------------------------------------
        eos = tokenizer.stoi['<eos>']
        pad = tokenizer.stoi['<pad>']

        ## fast version
        cache = [torch.empty((batch_size,0,decoder_dim), device=device) for i in range(num_layer)]
        for t in range(max_length-1):
            #last_token = token [:,:(t+1)]
            #text_embed = self.token_embed(last_token)
            #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

            last_token = token[:, t]
            text_embed = self.token_embed(last_token)
            text_embed = text_embed + text_pos[:,t] #
            text_embed = text_embed.reshape(batch_size,1,text_dim)

            x, cache = self.text_decode.forward_last(text_embed[:, -1:], cache, image_embed)
            x = x.reshape(batch_size,decoder_dim)

            l = self.logit(x)
            k = torch.argmax(l, -1)  # predict max
            token[:, t+1] = k

            if ((k == eos) | (k == pad)).all():  break

        predictions = token[:, 1:]
        return predictions      
'''