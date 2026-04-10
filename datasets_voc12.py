import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = '/data/djene/djene/MCTCon/voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform_img=None, transform_sal=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        if not os.path.exists(img_name_list_path):
            raise FileNotFoundError(f"Image list file not found: {img_name_list_path}")
        self.img_name_list = load_img_name_list(img_name_list_path)
        if not self.img_name_list:
            raise ValueError(f"Image list is empty: {img_name_list_path}")
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform_img = transform_img
        self.transform_sal = transform_sal
        self.train = train

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img_path = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        # if self.train:
            # Training: Return img, label, sal
        sal_path = os.path.join(self.voc12_root, 'SaliencyMaps', name + '.png')
        if not os.path.exists(sal_path):
            raise FileNotFoundError(f"Saliency map not found: {sal_path}")
        sal = Image.open(sal_path).convert("L")
        
        if self.transform_img and self.transform_sal:
            img = self.transform_img(img)
            sal = self.transform_sal(sal)
            # sal = (sal > 0.5).float().clone()  # Binarize and avoid non-resizable tensors
            sal = sal.round().clamp(0, 1)
        else:
            img = transforms.ToTensor()(img)
            sal = transforms.ToTensor()(sal).clone()
            sal = sal.round().clamp(0, 1)
            # sal = (sal > 0.5).float()
        
        return img, label, sal
        # else:
        #     # Validation: Return only img, label
        #     if self.transform_img:
        #         img = self.transform_img(img)
        #     else:
        #         img = transforms.ToTensor()(img)
            
        #     return img, label

    def __len__(self):
        return len(self.img_name_list)

class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=False, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"val" if train or gen_attn else "val"}_id.txt')
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
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
    transform_img, transform_sal = build_transform(is_train, args)  # Now returns a tuple
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform_img=transform_img, transform_sal=transform_sal)
        nb_classes = 20
    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform_img)
        nb_classes = 20

    return dataset, nb_classes


def build_transform(is_train, args):
    # Shared spatial transforms (applied to both image and saliency)
    shared_spatial = []
    if is_train:
        # Training: Random spatial augmentations
        shared_crop = transforms.RandomResizedCrop(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC)
        shared_flip = transforms.RandomHorizontalFlip(p=0.5)
        shared_spatial = [shared_crop, shared_flip]
    else:
        # Validation: Deterministic resizing and cropping
        # Skip resizing if gen_attention_maps, but ensure fixed size for consistency
        if not args.gen_attention_maps:
            shared_resize = transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC)
            shared_crop = transforms.CenterCrop(args.input_size)
            shared_spatial = [shared_resize, shared_crop]

    # Image-specific transforms (color jitter, normalize)
    img_specific = []
    if is_train:
        img_specific.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    img_specific += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]

    # Saliency-specific transforms (only ToTensor, no color jitter or normalize)
    sal_specific = [transforms.ToTensor()]

    # Compose the pipelines
    transform_img = transforms.Compose(shared_spatial + img_specific)
    transform_sal = transforms.Compose(shared_spatial + sal_specific)

    return transform_img, transform_sal