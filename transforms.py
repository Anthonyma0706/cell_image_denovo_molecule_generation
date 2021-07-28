from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from CFG import CFG
import albumentations as A

# transformations

# def get_transforms(*, data):
    
#     if data == 'train':
#         return Compose([
#             Resize(CFG.size, CFG.size),
#             HorizontalFlip(p=0.5),                  
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ShiftScaleRotate(p=0.5),   
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#             ToTensorV2(),
#         ])
    
#     elif data == 'valid':
#         return Compose([
#             Resize(CFG.size, CFG.size),
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#             ToTensorV2(),
#         ])

def get_train_transforms(CFG):
    return A.Compose([
            A.Resize(height=CFG.image_size, width=CFG.image_size,p=1.0),
            # A.Transpose(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
        #    A.RandomRotate90(p=0.5),
#             A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
#             A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.20,rotate_limit=20, p=0.5),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],p=1.0)

def get_val_transforms(CFG):
    return A.Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            #A.CenterCrop(height=CFG.image_size, width=CFG.image_size,p=1.0),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],p=1.0)
    
def get_test_transforms(CFG):
    return A.Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])