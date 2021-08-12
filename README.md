
# cell_image_denovo_molecule_generation

## De Novo hit-like molecule generation task using Cell Painting Assay (CPA) Cell Images


## Notes
This work builds with a CNN model and a transformer model, capturing the cell morphological features from the CPA 5-channel images via Transformer in transformer CNN then predicting/outputing the SMILES sequences using the Transformer Decoder.

![截屏2021-08-12 下午3 38 51](https://user-images.githubusercontent.com/57332047/129157211-3c29fea1-3fcc-464b-b92c-a46755b1c0ad.png)

Dataset & Background:
* Cell Image Library Dataset:Source: A dataset of images and morphological profiles of 30 000 small-molecule treatments using the Cell Painting assay. Bray et al.
(https://academic.oup.com/gigascience/article/6/12/giw014/2865213);
![background_project](https://user-images.githubusercontent.com/57332047/129158926-7e71e36e-8bd1-410a-9c90-37ddd74758da.png)
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













