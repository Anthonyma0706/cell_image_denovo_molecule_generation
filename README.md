# cell_image_denovo_molecule_generation

## De Novo hit-like molecule generation task using Cell Painting Assay (CPA) Cell Images


## Notes
This work builds with a CNN model and a transformer model, capturing the cell morphological features from the CPA 5-channel images via Transformer in transformer CNN then predicting/outputing the SMILES sequences using the Transformer Decoder.

![截屏2021-08-12 下午3 38 51](https://user-images.githubusercontent.com/57332047/129157211-3c29fea1-3fcc-464b-b92c-a46755b1c0ad.png)

### Dataset & Background:
* Cell Image Library Dataset:Source: A dataset of images and morphological profiles of 30 000 small-molecule treatments using the Cell Painting assay. Bray et al.
(https://academic.oup.com/gigascience/article/6/12/giw014/2865213);
![background_project_2](https://user-images.githubusercontent.com/57332047/129159124-4969f049-5d82-4d58-a313-fbbf9279343f.png)
![project_dataset_intro](https://user-images.githubusercontent.com/57332047/129157830-b784c295-8e6c-4607-873f-23fd071cc993.png)



## Requirements

This package requires:

* Python 3.8
* PyTorch (torch)
* PyTorch Lightning: pytorch_lightning
* timm
* [RDkit](http://www.rdkit.org/docs/Install.html)
* sklearn
* tqdm (for training Prior)
* pandas
* numpy
* cv2
* os
* albumentations
* moses

## Usage
#### Things you may need to acquaint:
* get to know about Cell Painting Assay and the CIL dataset
* /home/mingyang/mma/transformer_code/cil_data/data_process_group_5_organelles.ipynb: This shows how I clean and process the raw data, including reading the zipfile directly then grouping the images of the same well and plate and channels
* Note that you do not actually need to touch these data files. All files you need to use are introduced below, you may take a look to see the key attributes
#### Necessary data files/folders/paths on server 10.10.10:
- My own directory: /home/mingyang/mma/transformer_code
- /gxr/Omics/CIL/ : The directory on 10.10.10 where all the raw CIL cell images and processed npy files for 406 plates. Suggestion: Do not touch this folder, simply use the metadata containing the image file paths
- /home/mingyang/mma/transformer_code/split_data: This folder contains all the important dataset you will need fo this project
  ![截屏2021-08-12 下午4 45 27](https://user-images.githubusercontent.com/57332047/129167231-6a182fb0-3128-40e3-8eca-3b708c6a77a9.png)
  - trainset_by_cluster.csv : training set
  - testset_by_cluster.csv : test set
  - The npy file is a 5D array that derives from 5 original CIL image, corresponding to five channels as illustrating below
  ![截屏2021-08-12 下午4 56 33](https://user-images.githubusercontent.com/57332047/129168933-043eeeb5-7c56-49f0-b730-fcd683171741.png)


#### Important code files:
- CFG.py
- data.py
- transforms.py
- model.py
- train_transformer.py
- image_to_smiles.py
Import configurations:
All in **CFG.py**, detailed configs including:
* n_gpus
* batch_size
* epochs
* image_size, image_dim
* decoder_mode = 'transformer'
* encoder_name = 'tnt_s_patch16_224'
* n_fold = 5, trn_fold = [0]
* train_folds_path = './split_data/trainset_by_cluster.csv' : a data path file with SMILES sequence label, npy_file_path, fold_number (0-4): for cross-validation during training. Fo more details please view this file
* tokenizer_path = './commercial_data/tokenizer_smiles_new.pth': a .pth file of the dictionary used for transformer, generated using test_smiles_token.py
* cDNA_testset = False
* greedy_sample = True # whether predicting/inferecing using greedy, rather than Multinomial Sampling
* multinomial_sample_size = 0, usually 200 for Multinomial sampling
* image_channels = 5 (channels to use 5 or 3 or 1)
* channel = '' (used fo 1 channel 'Hoechst' , 'ERSyto', 'ERSytoBleed')


### Usage Pipeline:
1. Data processing:
* Split the training & test set using fingerprint clusters: /home/mingyang/mma/transformer_code/split_data_generate_input.ipynb

2. Training:
* Note: get the pretrained weights for CNN (now the default weight is for tnt_s_patch16_224):
   - Way 1: use my weight at /home/mingyang/.cache/torch/hub/checkpoints and copy to your path ~/.cache/torch/hub/checkpoints
   - Way 2: download the pretrained weights from github: (The link will be output on the screen the first time you run train_transformer.py)
then save the pretrained weights to ~/.cache/torch/hub/checkpoints (Please be careful that ~/ should be updated to the directory where your anaconda is installed)
* config train_transformer.py: adjust the GPU at the bottom, choose a free one using command `nvidia-smi` on the server; load the weights
![截屏2021-08-12 下午5 14 30](https://user-images.githubusercontent.com/57332047/129171422-e9906edf-80db-4380-aa33-b821492521f1.png)

* run `python train_transformer.py` (adjust CFG.py beforehand)
* WEIGHTS will be saved at ./model![截屏2021-08-12 下午5 17 40](https://user-images.githubusercontent.com/57332047/129171993-1508b3e4-39b9-4d3f-b2dc-4d56b48178e6.png) 
* Please note that the weight will be saved with name **last.ckpt** at the last epoch, please **DO RENAME** each one after training! Recommand rename using your specific task, date and score.

3. Prediction: 
* ![截屏2021-08-12 下午5 20 25](https://user-images.githubusercontent.com/57332047/129172409-3bb04941-5b94-43bb-bb83-30c810fe5aa4.png)
* run `python image_to_smiles.py` (adjust CFG.py and GPU at the TOP of the file beforehand)

4. Evaluation (Metrics):
View the jupyter notebook .ipynb files such as:
- testset_eval_new.ipynb
- eval_cDNA_MS_predictions.ipynb
- ...










