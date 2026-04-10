
#Import the required libraries and modules
import numpy as np 
import pandas as pd
# from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
import cv2
import albumentations as A
import os

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

class CreateDataset(Dataset):
    
    def __init__(self, img_name_list_path, img_root, mask_path, mean, std, train=True, transform=None):
        self.img_root = img_root
        self.mask_path = mask_path
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_val" if train else "test"}.txt')   #Modified
        self.img_name_list = load_img_name_list(img_name_list_path)
        # self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        
    # def __len__(self):
    #     return len(self.X)
    
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = cv2.imread(os.path.join(self.img_root, 'JPEGImages', name + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, name + '.png'), cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = transforms.Compose([transforms.ToTensor(), transforms .Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        
        return img, mask
    
    def __len__(self):
        return len(self.img_name_list)
    

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def build_dataset(is_train, args):
    transform = data_aug(is_train)
    dataset = None
    nb_classes = None
    dataset = CreateDataset(img_name_list_path=args.img_list, img_root=args.data_path, mask_path=args.mask_path, mean=mean, std=std,train=is_train, transform=transform)
    nb_classes = 4
    return dataset, nb_classes

def data_aug(is_train):
    if is_train:
         t = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(p=0.4), A.VerticalFlip(p=0.4), 
                     A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.Blur(p = 0.3),
                     A.RandomRotate90(p=0.3),
                     A.GaussNoise(p = 0.3)])
    else:
        t = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_NEAREST)])
    return t

