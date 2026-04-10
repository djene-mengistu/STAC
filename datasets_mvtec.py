import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# from torchvision import transforms as Tt


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = '/data/djene/djene/MCTCon/mvtec/leather/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.png'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

# rfh = transforms.RandomHorizontalFlip(p=0.3)
# rfv = transforms.RandomVerticalFlip(p=0.3)

class NEUDataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform=None, gen_attn=False):
        # if train:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train"}.txt')
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"test"}.txt')
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "test"}.txt')   ##The original
        img_name_list_path = os.path.join(img_name_list_path, f'{"all_images" if train else "def_images"}.txt')   #Modified
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'rJPEGImages', name + '.png')).convert("RGB")
        sal = PIL.Image.open(os.path.join(self.voc12_root, 'rSALmapsALL', name + '.png'))#.convert("L") # 'convert 'RGB' changes the saliency map to 3 channel
        # sal = sal / 255.0 #Convert to zeros and 1
        # sal_mask = Image.open(mask_path).convert('L')
        label = torch.from_numpy(self.label_list[idx])


        # Data transformation
        
        sal_transform = transforms.Compose([
            # rfh,
            # rfv,
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  # Resize to match the input size expected by the backbone
            transforms.ToTensor()
        ])

        if self.transform:
             
            img = self.transform(img)
            sal = sal_transform(sal)
            # aug = self.transform(image=img, mask=sal)
            # img = aug['image']
            # sal = aug['mask'].float()
            # sal = np.asarray(sal)
            # # sal = torch.from_numpy(sal)
            
            # # sal = sal[0:1, :, :] #to keep only single channel for the saliency map
            # sal = sal/255. #To change the saliency map between [0 and 1]
            # # sal = torch.mean(sal, dim=0, keepdim=True)
            # sal = torch.from_numpy(sal)
            sal = sal.round().clamp(0, 1)

        return img, label, sal  #Include sal if saliency

    def __len__(self):
        return len(self.img_name_list)


class NEUDatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=False, transform=None, gen_attn=True, unit=1):
        # if train:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train"}.txt')
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"test"}.txt')
        img_name_list_path = os.path.join(img_name_list_path, f'{"all_images" if train else "def_images"}.txt')  #Modified
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'rJPEGImages', name + '.png')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.BICUBIC) 
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


def build_dataset(is_train, args, gen_attn=False):
    transform = build_transform(is_train, args)
    dataset = None
    nb_classes = None

    if args.data_set == 'mvtec_seg':
        dataset = NEUDataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 1
    elif args.data_set == 'mvtec_seg_MS':
        dataset = NEUDatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 1
    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]

         # Create an empty list for transforms
        transform_list = []

        # Example: Add random horizontal flip with probability 0.5
        # transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Add other transformations as needed
        # transform_list.append(rfh)
        # transform_list.append(rfv)
        transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        
        # Replace Resize and possibly RandomCrop based on conditions
        # if not resize_im:
        #     transform_list.append(transforms.RandomCrop(args.input_size, padding=4))
        # else:
        transform_list.append(transforms.Resize(args.input_size, interpolation=3))
        # transform_list.append(transforms.CenterCrop(args.input_size))

        # # Add common transformations
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        # Create empty lists for transformations
        # transform_img_list = []
        # transform_sal_list = []

        # # Example: Add random horizontal flip with probability 0.5
        # rf = transforms.RandomHorizontalFlip(p=0.5)
        # transform_img_list.append(rf)
        # transform_sal_list.append(rf)

        # # Add other transformations as needed
        # transform_img_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
        # transform_sal_list.append(transforms.Resize(args.input_size, interpolation=3))

        # # Define separate transformations for image
        # transform_img_list.append(transforms.ToTensor())
        # transform_img_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

        # # Define separate transformations for saliency map
        # transform_sal_list.append(transforms.ToTensor())

        # transform_img = transforms.Compose(transform_img_list)
        # transform_sal = transforms.Compose(transform_sal_list)
        
        # Compose all transforms
        transform = transforms.Compose(transform_list)
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     # auto_augment=args.aa,
        #     interpolation=args.train_interpolation,
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        # )
        # if not resize_im:
        #     # replace RandomResizedCropAndInterpolation with
        #     # RandomCrop
        #     transform.transforms[0] = transforms.RandomCrop(
        #         args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        # size = int((256 / 224) * args.input_size) #newly commented out
        t.append(
            transforms.Resize(args.input_size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            # A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_NEAREST)])
        )
        # t.append(transforms.CenterCrop(args.input_size)) #No, cropping
    # T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    t.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return transforms.Compose(t)
