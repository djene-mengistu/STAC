import os
import time
import imageio
import argparse
import importlib
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from torch.multiprocessing import Process

from util import imutils, pyutils
from util.imutils import HWC_to_CHW
from network.resnet38d import Normalize
# from metadata.neu_dataset import load_img_id_list, load_img_label_list_from_npy
from metadata.mvtec_dataset import load_img_id_list, load_img_label_list_from_npy


start = time.time()


def overlay_cam_on_image(image, cam, colormap='jet'):
    """
    Overlay CAM heatmap on original image.
    image: np.array (H, W, 3), uint8, [0, 255]
    cam: np.array (H, W), float32, any range (will be normalized)
    Returns: blended image (H, W, 3), uint8
    """
    import matplotlib.pyplot as plt

    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)  # normalize to [0,1]

    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    blended = (0.5 * image + 0.5 * heatmap).astype(np.uint8)
    return blended


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--infer_list", default="mtd_seg/test.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", nargs='*', type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)
    parser.add_argument("--img_root", default='/home/dengjie/djene/data/MTD_VOC', type=str)
    parser.add_argument("--tasks", choices=['train', 'test'], default='train', type=str)
    parser.add_argument("--crf", default=None, type=str)
    parser.add_argument("--crf_alpha", default=(4,32), type=int, nargs='+')
    parser.add_argument("--crf_t", default=(5,10), type=int, nargs='+')
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--cam_png", default=None, type=str)  # kept for compatibility, but unused in cls mode
    parser.add_argument("--cam_vis", default=None, type=str)   # NEW: for CAM visualizations
    parser.add_argument("--thr", default=0.50, type=float)
    parser.add_argument("--dataset", default='mtd_seg', type=str)
    args = parser.parse_args()

    if args.dataset == 'neu_seg':
        args.num_classes = 3
    elif args.dataset == 'dagm_seg':
        args.num_classes = 6
    elif args.dataset == 'mtd_seg':
        args.num_classes = 5
    elif args.dataset == 'mvtec_seg':
        args.num_classes = 1
    else:
        raise Exception('Unsupported dataset')

    # Model type
    if 'cls' in args.network:
        args.network_type = 'cls'
        args.model_num_classes = args.num_classes
    elif 'eps' in args.network:
        args.network_type = 'eps'
        # args.model_num_classes = args.num_classes + 1
        args.model_num_classes = args.num_classes
    else:
        raise Exception('Network must contain "cls" or "eps"')

    # Ensure we are in cls mode for this script (as per your request)
    # if args.network_type != 'cls':
    #     raise ValueError("This visualization script is intended for 'cls' mode only.")

    # Create output directories
    args.save_type = []
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)
    if args.cam_vis is not None:
        os.makedirs(args.cam_vis, exist_ok=True)
        args.save_type.append(args.cam_vis)
    if args.crf is not None:
        args.crf_list = []
        for t in args.crf_t:
            for alpha in args.crf_alpha:
                crf_folder = os.path.join(args.crf, f'crf_{t}_{alpha}')
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(crf_folder)

    # Process counts
    if args.n_processes_per_gpu is None:
        args.n_processes_per_gpu = [args.n_total_processes]
    else:
        args.n_processes_per_gpu = [int(x) for x in args.n_processes_per_gpu]
    args.n_total_processes = sum(args.n_processes_per_gpu)
    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    multi_scale_image_list = []
    multi_scale_flipped_image_list = []

    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.BICUBIC)
        multi_scale_image_list.append(scaled_image)

    for i in range(len(multi_scale_image_list)):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])

    for i in range(len(multi_scale_image_list)):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())

    return multi_scale_flipped_image_list


def predict_cam(model, image, label, gpu, network_type):
    original_image_size = np.asarray(image).shape[:2]
    scales = (1.0, 1.5)
    normalize = Normalize()
    transform = torchvision.transforms.Compose([np.asarray, normalize, HWC_to_CHW])
    multi_scale_flipped_image_list = preprocess(image, scales, transform)

    cam_list = []
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(multi_scale_flipped_image_list):
            img_tensor = torch.from_numpy(img).unsqueeze(0).cuda(gpu)
            cam = model.forward_cam(img_tensor)  # Should return (1, C, H, W)

            cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]
            cam = cam.cpu().numpy() * label.reshape(args.num_classes, 1, 1)

            if i % 2 == 1:  # flipped
                cam = np.flip(cam, axis=-1)
            cam_list.append(cam)

    return cam_list


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = {0: crf_score[0]}
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]
    return n_crf_al


def infer_cam_mp(process_id, image_ids, label_list, cur_gpu):
    print(f'Process {os.getpid()} starts on GPU {cur_gpu}...')

    model = getattr(importlib.import_module(args.network), 'Net')(args.model_num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=f'cuda:{cur_gpu}'))
    model = model.cuda(cur_gpu)
    model.eval()

    for i, (img_id, label) in enumerate(zip(image_ids, label_list)):
        img_path = os.path.join(args.img_root, img_id + '.png')
        img = Image.open(img_path).convert('RGB')
        org_img = np.asarray(img)

        cam_list = predict_cam(model, img, label, cur_gpu, args.network_type)
        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, axis=(1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for j in range(args.num_classes):
            if label[j] > 1e-5:
                cam_dict[j] = norm_cam[j]

        # Save raw CAMs (optional)
        if args.cam_npy is not None:
            np.save(os.path.join(args.cam_npy, img_id + '.npy'), cam_dict)

        # Save CAM visualizations
        if args.cam_vis is not None:
            for cls_idx in cam_dict:
                overlay = overlay_cam_on_image(org_img, cam_dict[cls_idx])
                vis_path = os.path.join(args.cam_vis, f"{img_id}_cls{cls_idx}.png")
                imageio.imwrite(vis_path, overlay)

        # CRF (optional)
        if args.crf is not None:
            for folder, t, alpha in args.crf_list:
                cam_crf = _crf_with_alpha(org_img, cam_dict, alpha, t=t)
                np.save(os.path.join(folder, img_id + '.npy'), cam_crf)

        if i % 10 == 0:
            print(f'PID {process_id}, {i}/{len(image_ids)} done')


def main_mp():
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset, args.tasks)
    assert len(image_ids) == len(label_list)

    # Skip already processed
    if args.save_type:
        saved_list = set()
        for save_dir in args.save_type:
            if os.path.exists(save_dir):
                saved_list.update({f[:-4] for f in os.listdir(save_dir) if f.endswith('.npy') or f.endswith('.png')})
        if saved_list:
            new_image_ids, new_label_list = [], []
            for img_id, label in zip(image_ids, label_list):
                if img_id not in saved_list:
                    new_image_ids.append(img_id)
                    new_label_list.append(label)
            image_ids, label_list = new_image_ids, new_label_list

    n_total_images = len(image_ids)
    n_total_processes = args.n_total_processes

    print('===========================')
    print('OVERALL INFORMATION')
    print('n_gpus:', args.n_gpus)
    print('n_processes_per_gpu:', args.n_processes_per_gpu)
    print('n_total_processes:', n_total_processes)
    print('n_total_images:', n_total_images)
    print('n_images_to_proceed:', len(image_ids))
    print('===========================')

    if len(image_ids) == 0:
        print("All images already processed. Exiting.")
        return

    # Split data
    sub_image_ids = []
    sub_label_list = []
    split_size = len(image_ids) // n_total_processes
    for i in range(n_total_processes):
        start_idx = i * split_size
        end_idx = None if i == n_total_processes - 1 else (i + 1) * split_size
        sub_image_ids.append(image_ids[start_idx:end_idx])
        sub_label_list.append(label_list[start_idx:end_idx])

    # Assign GPUs
    gpu_list = []
    for gpu_id, num_proc in enumerate(args.n_processes_per_gpu):
        gpu_list.extend([gpu_id] * num_proc)

    # Launch processes
    processes = []
    for idx in range(n_total_processes):
        p = Process(target=infer_cam_mp, args=(idx, sub_image_ids[idx], sub_label_list[idx], gpu_list[idx]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    global args
    args = parse_args()

    main_mp()
    print(f"Total time: {time.time() - start:.2f} seconds")