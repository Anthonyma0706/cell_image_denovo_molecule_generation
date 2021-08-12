
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
#### Important data files/folders/paths on server 10.10.10:
- My own directory: /home/mingyang/mma/transformer_code
- /gxr/Omics/CIL/ : The directory on 10.10.10 where all the raw CIL cell images and processed npy files for 406 plates. Suggestion: Do not touch this folder, simply use the metadata containing the image file paths
- /home/mingyang/mma/transformer_code/split_data: This folder contains all the important dataset you will need fo this project
  ![截屏2021-08-12 下午4 45 27](https://user-images.githubusercontent.com/57332047/129167231-6a182fb0-3128-40e3-8eca-3b708c6a77a9.png)
  - trainset_by_cluster.csv : training set
  - testset_by_cluster.csv : test set
  - The npy file is a 5D array that derives from 5 original CIL image, corresponding to five channels as illustrating below
  ![截屏2021-08-12 下午4 53 20](https://user-images.githubusercontent.com/57332047/129168784-e47a0bd7-203b-4a00-bd49-ccc2335424f4.png)


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

1. Training:
* download then save the pretrained weights to ~/.cache/torch/hub/checkpoints (Please be careful that ~/ should be updated to the directory where your anaconda is installed) Also please download the weight on the github link (will be output on the screen the first time you run train_transformer.py)
* run train_transformer.py (adjust CFG.py beforehand)

2. Prediction:
* run image_to_smiles.py (adjust CFG.py beforehand)

3. Evaluation (Metrics):
View the jupyter notebook .ipynb files such as:
- testset_eval_new.ipynb
- eval_cDNA_MS_predictions.ipynb
- ...










