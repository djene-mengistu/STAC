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
    """Dataset class for VOC12 images and saliency maps."""
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

        sal_path = os.path.join(self.voc12_root, 'SaliencyMaps', name + '.png')
        if not os.path.exists(sal_path):
            raise FileNotFoundError(f"Saliency map not found: {sal_path}")
        sal = Image.open(sal_path).convert("L")
        
        if self.transform_img and self.transform_sal:
            # Ensure same random seed for training to align image and saliency transforms
            if self.train:
                seed = torch.random.initial_seed()
                torch.manual_seed(seed)
                img = self.transform_img(img)
                torch.manual_seed(seed)  # Reset seed for saliency
                sal = self.transform_sal(sal)
            else:
                img = self.transform_img(img)
                sal = self.transform_sal(sal)#.contiguous()  # Ensure contiguous tensor
            sal = sal.round().clamp(0, 1).contiguous()  # Ensure contiguous tensor  # Binarize saliency map
        else:
            img = transforms.ToTensor()(img)
            sal = transforms.ToTensor()(sal).contiguous()  # Ensure contiguous tensor
            sal = sal.round().clamp(0, 1).contiguous()  # Ensure contiguous tensor
        
        return img, label, sal

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
    """Build separate transforms for images and saliency maps."""
    resize_im = args.input_size > 32
    
    # Image transformation
    image_transform = []
    if is_train:
        # Training: Use timm's create_transform for images
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        image_transform = transform
    else:
        # Validation: Resize and crop for images
        if resize_im and not args.gen_attention_maps:
            size = int((256 / 224) * args.input_size)
            image_transform.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
            )
            image_transform.append(transforms.CenterCrop(args.input_size))
        image_transform.append(transforms.ToTensor())
        image_transform.append(transforms.Normalize(
            IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        image_transform = transforms.Compose(image_transform)

    # Saliency map transformation
    saliency_transform = []
    if is_train:
        # Training: Apply same spatial transforms as image for alignment
        spatial_transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=0,  # No color jitter for saliency
            # auto_augment=None,  # No auto-augment for saliency
            interpolation=args.train_interpolation,
            # re_prob=0,  # No random erasing for saliency
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            spatial_transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        # Only include spatial transforms (e.g., RandomResizedCrop or RandomCrop)
        saliency_transform = [
            t for t in spatial_transform.transforms
            if isinstance(t, (transforms.RandomResizedCrop, transforms.RandomCrop))
        ]
    else:
        # Validation: Apply same spatial transforms as image
        if resize_im and not args.gen_attention_maps:
            size = int((256 / 224) * args.input_size)
            saliency_transform.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
            )
            saliency_transform.append(transforms.CenterCrop(args.input_size))
    
    # Convert saliency map to tensor
    saliency_transform.append(transforms.ToTensor())
    
    # Optional: Normalize saliency map if not in [0, 1]
    # Assuming .png saliency maps are in [0, 255], ToTensor scales to [0, 1]
    # If further normalization is needed, uncomment:
    # saliency_transform.append(transforms.Normalize(mean=(0,), std=(255,)))
    
    saliency_transform = transforms.Compose(saliency_transform)
    
    return image_transform, saliency_transform