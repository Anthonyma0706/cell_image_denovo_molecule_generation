import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)  ###OpenCV与Pytorch互锁的问题，关闭OpenCV的多线程即可解决问题
import albumentations as A
import os
# train = pd.read_pickle("./inchi_data/train.pkl")

# def get_train_file_path(image_id):
#     return "/gxr/fanyang/Molecular-translation/train/{}/{}/{}/{}.png".format(
#         image_id[0], image_id[1], image_id[2], image_id 
#     )

# train['file_path'] = train['image_id'].apply(get_train_file_path)
# # print(f'train.shape: {train.shape}')
# # print(train)

# train.to_pickle('./inchi_data/train_new.pkl')
# train = pd.read_pickle("./inchi_data/train_new.pkl")
# # print(train)
#print()

class Tokenizer(object):
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)
    '''
    def update_stoi_itos(self):
        # update stoi, add '7' to vocab
        update = False
        new_stoi = {}
        for s in self.stoi:
            if s == '6':
                new_stoi[s] = self.stoi[s]
                new_stoi['7'] = 13
                update = True
            elif update:
                new_stoi[s] = self.stoi[s] + 1
            else:
                new_stoi[s] = self.stoi[s] 
               
        self.stoi = new_stoi  
        
        
        update = False
        new_itos = {}
        for i in self.itos:
            if i < 13:
                new_itos[i] = self.itos[i]
            elif i == 13:
                new_itos[i] = '7' # insert 7
            else:
                new_itos[i] = self.itos[i-1]

        new_itos[53] = '<pad>'

        self.itos = new_itos
        #print('===============================VOCAB UPDATED========================================')
        
        
      '''  
        
        
     
    
    def fit_on_texts(self, texts):
        
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>') # start
        vocab.append('<eos>') # end
        vocab.append('<pad>') # adding
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        
        sequence = []
        sequence.append(self.stoi['<sos>'])
        #print(self.stoi)
        #print(self.itos)
        for s in text.split(' '):
            sequence.append(self.stoi[s])
            #try:
            #    sequence.append(self.stoi[s])
            #except KeyError:
            #    print('WARNING: The key : '+ s+ ' does not exist in SMILES VOCAB!')
            #    continue
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
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption
    
    def predict_captions(self, sequences):
        
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions
        

    def one_predict_to_inchi(self, predict):
        
        # inchi = 'InChI=1S/'
        inchi = ''
        #print(type(predict))
       # print(predict)
        print(predict.shape)
        if predict.shape == (1, 96):
            predict = predict[0]
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi
    
    def predict_to_inchi(self, predict):
        
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, CFG, df, tokenizer, transform=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.tokenizer = tokenizer
        #self.labels = df['smiles_unique_text'].values
        self.labels = df['space_smiles'].values # 更改命名
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ### 现在先直接用一张图片test一下
        #image_id = self.df.iloc[idx]['image_id']
        # image_path = os.path.join(self.CFG.train_dir,image_id[0],image_id[1],image_id[2], f'{image_id}.png')
        #image_path = self.df.iloc[idx]['new_file_path']
        
        
        # image_path = os.path.join(self.CFG.train_dir,image_id[0],image_id[1],image_id[2], f'{image_id}.png')
        #image_path = self.df.iloc[idx]['new_file_path']
        
        #image_path = 'SMILES_test_1.png'
        image_path = self.df.iloc[idx]['npy_path'] # get the 绝对路径 string directly
        #print(image_path)
        #print(image_path)
        #image = cv2.imread(image_path, 0)
        image_5D = np.load(image_path) # (5, 520, 696)
        #image = image / 15 # divide by 15 since the max pixels are all 15  <- suggested by Jixian
        '''
        if self.CFG.image_channels == 3:   
            image_3D = image_5D[:3,:,:]

            image_3D_new = np.copy(image_3D) # exchange rows
            image_3D_new[0,:,:] = image_3D[2,:,:] # Hoechst is first
            image_3D_new[1,:,:] = image_3D[0,:,:] # ERsyto is second
            image_3D_new[2,:,:] = image_3D[1,:,:] # ERSytoBleed is third

            #print(image_3D_new[0].shape)
            #print(image_3D_new[0,:,:].shape)
            image_5D = image_3D_new
        '''    
        if self.CFG.image_channels == 1:   
            image_3D = image_5D[:3,:,:]

            image_3D_new = np.copy(image_3D) # exchange rows
            image_3D_new[0,:,:] = image_3D[2,:,:] # Hoechst is first
            image_3D_new[1,:,:] = image_3D[0,:,:] # ERsyto is second
            image_3D_new[2,:,:] = image_3D[1,:,:] # ERSytoBleed is third

            #print(image_3D_new[0].shape)
            #print(image_3D_new[0,:,:].shape)
            #image_5D = image_3D_new[0,:,:]  # We TRAIN Hoechst FIRST
            #image_5D = image_3D_new[1,:,:]  # We TRAIN ERsyto SECOND
            image_5D = image_3D_new[2,:,:]  # We TRAIN ERsytoBleed THIRD
            image_5D = np.expand_dims(image_5D,axis=0) # ADD another dimension when using 1 channel
            #image_5D = torch.unsqueeze(x, 0)
         

        # print(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            image_5D = image_5D.transpose(1,2,0)
            augmented = self.transform(image=image_5D)
            image = augmented['image']
            image = image/15
            
            #print(image.shape)
            
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        #print('=====================================IMAGE PROCESSED ======================================')
        return torch.tensor(image, dtype = torch.float), torch.LongTensor(label), label_length
    

    
    
    
class ValidDataset(Dataset):
    def __init__(self, CFG, df, transform=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        
        self.labels = df['smiles'].values # 更改命名
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #image_id = self.df.iloc[idx]['image_id']
        # image_path = os.path.join(self.CFG.train_dir,image_id[0],image_id[1],image_id[2], f'{image_id}.png')
        #image_path = 'SMILES_test_1.png'
        #image_path = self.df.iloc[idx]['new_file_path']
        #image = cv2.imread(image_path)
        
        image_path = self.df.iloc[idx]['npy_path'] # get the 绝对路径 string directly
        #image_5D = cv2.imread(image_path)  # when do the cDNA prediction
        image_5D = np.load(image_path) # (5/3, 520, 696)
        #print(image_5D.shape) # -> should be 1080,1080, 3  for cDNA testset
        
        
        if self.CFG.image_channels == 1:   
            image_3D = image_5D[:3,:,:]

            image_3D_new = np.copy(image_3D) # exchange rows
            image_3D_new[0,:,:] = image_3D[2,:,:] # Hoechst is first
            image_3D_new[1,:,:] = image_3D[0,:,:] # ERsyto is second
            image_3D_new[2,:,:] = image_3D[1,:,:] # ERSytoBleed is third

            #image_5D = image_3D_new[0,:,:]  # We TEST Hoechst FIRST
            #image_5D = image_3D_new[1,:,:]  # We TEST ERsyto SECOND
            image_5D = image_3D_new[1,:,:]  # We TEST ERSytoBleed Third
    
            image_5D = np.expand_dims(image_5D,axis=0) # ADD another dimension when using 1 channel

        
        if self.transform:
            image_5D = image_5D.transpose(1,2,0) 
            augmented = self.transform(image=image_5D)
            image = augmented['image']
            image = image/15
            #print(image.shape)
        label = self.labels[idx] ## text labels
        #print('=================================IMAGE PROCESSED ======================================')
        #sourceTensor.clone().detach()
        return torch.tensor(image,dtype = torch.float) , label
   
    
    
    
    
    
class TestDataset(Dataset):
    def __init__(self, CFG, df, transform=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        #self.labels = df['smiles_unique_text'].values
        #self.labels = df['smiles'].values # 更改命名
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        #image = cv2.imread(image_path)
        
        image_path = self.df.iloc[idx]['npy_path'] # get the 绝对路径 string directly
        image_5D = cv2.imread(image_path)  # when do the cDNA prediction
        
        #image_5D = np.load(image_path) # (5/3, 520, 696)
        #print(image_5D.shape) # -> should be 1080,1080, 3  for cDNA testset
        if self.transform:
            #image_5D = image_5D.transpose(1,2,0)  # comment when using cDNA testset
            #print(image_5D.shape)
            augmented = self.transform(image=image_5D)
            image = augmented['image']
            image = image/15
            if self.CFG.image_channels == 1:  
                #print('CFG channels is 1')
                #image = image[0] # for Hoechst
                image = image[1] # for ER syto
                image = torch.reshape(image, (1,224,224))
                
        #label = self.labels[idx] ## text labels
        #print('=================================IMAGE PROCESSED ======================================')
        #sourceTensor.clone().detach()
        return torch.tensor(image,dtype = torch.float) #, label

# class TestDataset(Dataset):
#     def __init__(self, df, transform=None):
#         super().__init__()
#         self.df            = df
#         self.file_paths    = df['file_path'].values
#         self.transform     = transform
#         self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         h, w, _ = image.shape
#         if h > w:
#             image = self.fix_transform(image=image)['image']
#         if self.transform:
#             augmented = self.transform(image=image)
#             image     = augmented['image']
#         return image