import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils
from pico_loss import compute_pico_loss, label_onehot  
# from utils import imutils, pyutils 
import PIL

from skimage.metrics import mean_squared_error #For saleincy metrics
from skimage import io, img_as_float
from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

def cam_labels(cams, targets, num_classes=3):
    cls_attentions = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
    # at_gt = cls_attentions* (targets.view(cls_attentions.shape[0], 3, 1, 1).expand(cls_attentions.shape[0], 3, 224, 224))

    batch_predict = []
    for b in range(cams.shape[0]):
        if (targets[b].sum()) > 1e-5:
            cam_dict = {}
            for cls_ind in range(num_classes):
                if targets[b, cls_ind] > 0:
                    cls_attention = cls_attentions[b, cls_ind, :]
                    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                    cam_dict[cls_ind] = cls_attention                    
            h, w = list(cam_dict.values())[0].shape
            tensor = torch.zeros((num_classes, h, w), dtype=torch.float32, device=cams.device)
            for key in cam_dict.keys():
                    tensor[key] = cam_dict[key] 
            predict = torch.sum(tensor, dim=0)                     
            batch_predict.append(predict)

    batch_predict = torch.stack(batch_predict, dim=0)
    batch_predict = batch_predict.unsqueeze(1)
    return batch_predict

# def pico_inputs(cams, targets, weak_threshold, num_classes):
#     cls_attentions = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
     
#     with torch.no_grad():
#             probs = cls_attentions
#             mask = probs.ge(weak_threshold).float()
#             mask = mask.unsqueeze(1)

#             batch_predict = []
#             for b in range(cams.shape[0]):
#                 if (targets[b].sum()) > 1e-5:
#                     cam_dict = {}
#                     for cls_ind in range(num_classes):
#                         if targets[b, cls_ind] > 0:
#                             cls_attention = cls_attentions[b, cls_ind, :]
#                             cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
#                             cam_dict[cls_ind] = cls_attention                    
#                     h, w = list(cam_dict.values())[0].shape
#                     tensor = torch.zeros((num_classes, h, w), dtype=torch.float32, device=cams.device)
#                     for key in cam_dict.keys():
#                             tensor[key] = cam_dict[key]           
#                     batch_predict.append(tensor)

#             label = torch.stack(batch_predict, dim=0)
#             # batch_predict = batch_predict.unsqueeze(1)
#     return probs, mask, label
def pico_inputs(cams, sal, targets, weak_threshold, num_classes):
    bcg = 1 - sal #Compute the saleincy map
    cls_attentions = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False) 

    batch_predict = []
    for b in range(cams.shape[0]):
        if (targets[b].sum()) > 1e-5:
            cam_dict = {}
            for cls_ind in range(num_classes):
                if targets[b, cls_ind] > 0:
                    cls_attention = cls_attentions[b, cls_ind, :]
                    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                    cam_dict[cls_ind] = cls_attention                    
            h, w = list(cam_dict.values())[0].shape
            tensor = torch.zeros((num_classes + 1, h, w), dtype=torch.float32, device=cams.device)
            for key in cam_dict.keys():
                    tensor[key + 1] = cam_dict[key]
            tensor[0,:] = bcg[b] #Setting the background from the saliency map                  
            batch_predict.append(tensor)

    probs = torch.stack(batch_predict, dim=0)
    logits, label = torch.max(probs, dim=1)
    mask = logits.ge(weak_threshold).float()
    mask = mask.unsqueeze(1) #Change this value accordingly
    labels = label_onehot(label, 4) #Change accordingly (4 NEU, 6 MTD, 7 DAGM, 21 VOC12)
    return probs, mask, labels


##For the MVTEC dataset
def cam_labels_pos(cams, targets, output_size=(224, 224)):
    """
    Generate binary attention maps from 1-channel CAMs.
    Output shape: (B, 1, H, W)
    """
    cls_attentions = F.interpolate(cams, size=output_size, mode='bilinear', align_corners=False)
    batch_predict = []
    for b in range(cams.shape[0]):
        if targets[b] > 1e-5:
            cam_fg = cls_attentions[b, 0]
            cam_fg = (cam_fg - cam_fg.min()) / (cam_fg.max() - cam_fg.min() + 1e-8)
            batch_predict.append(cam_fg)
        else:
            batch_predict.append(torch.zeros_like(cls_attentions[b, 0]))
    return torch.stack(batch_predict, dim=0).unsqueeze(1)

def cam_labels_neg(cams, targets, output_size=(224, 224)):
    """
    Generate binary attention maps from 1-channel CAMs.
    Output shape: (B, 1, H, W)
    """
    cls_attentions = F.interpolate(cams, size=output_size, mode='bilinear', align_corners=False)
    batch_predict = []
    for b in range(cams.shape[0]):
        if targets[b] < 0.5:
            cam_fg = cls_attentions[b, 0]
            cam_fg = (cam_fg - cam_fg.min()) / (cam_fg.max() - cam_fg.min() + 1e-8)
            batch_predict.append(cam_fg)
        # else:
        # batch_predict.append(torch.zeros_like(cls_attentions[b, 0]))
    return torch.stack(batch_predict, dim=0).unsqueeze(1)

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
   
    for samples, targets, sal in metric_logger.log_every(data_loader, print_freq, header):  

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sal = sal.to(device, non_blocking=True) 

        patch_outputs = None
        c_outputs = None
        with torch.cuda.amp.autocast():
            outputs, cams1, cams2, proj = model(samples)
            if len(outputs) == 2:
                outputs, patch_outputs = outputs
            elif len(outputs) == 3:
                outputs, c_outputs, patch_outputs = outputs           
            

            #Classification loss
            loss_cls = F.multilabel_soft_margin_loss(outputs, targets) #modified from loss -->loss_cls
            
            # # SALIENCY LOSS
            # batch_fg_1 = cam_labels_pos(cams1, targets)   #for MVTEC dataset 
            # batch_fg_2 = cam_labels_pos(cams2, targets)   #for MVTEC dataset
            # loss_sal = F.mse_loss(batch_fg_1, sal) +  F.mse_loss(batch_fg_2, sal)     

            batch_predict_1 = cam_labels(cams1, targets, 3)   #Generate the CAM               
            batch_predict_2 = cam_labels(cams2, targets, 3)

            loss_sal = F.mse_loss(batch_predict_1, sal) +  F.mse_loss(batch_predict_2, sal)
             
            # REGIONAL CONTRASTIVE LOSS 
            probs, mask, label = pico_inputs(cams1, sal, targets, 0.5, 3) #To compute the loss for the pixel-level contrastive loss       

            loss_pico = compute_pico_loss(proj, label, mask, probs, args.strong_thresh, args.temp, args.num_queries, args.num_negatives)

            metric_logger.update(mct_loss=loss_cls.item())
            metric_logger.update(sal_loss=loss_sal.item()) #For the saliency map
            metric_logger.update(pico_loss=loss_pico.item()) #For the saliency map 
            loss = loss_cls + 2.0*loss_sal + 0.2*loss_pico #The coefficient of the loss functions Ablation con{0.1,0.2,0.3,0.4, 0.5} and sal{0.5,1,2,3}

            if c_outputs is not None:
                # c_outputs = c_outputs[:,:,:-1,:] #DON'T forget to change this
                c_outputs = c_outputs[-args.num_cct:]
                output_cls_embeddings = F.normalize(c_outputs, dim=-1)  # 12xBxCxD --->12*B*3*384
                scores = output_cls_embeddings @ output_cls_embeddings.permute(0, 1, 3, 2)  # 12xBxCxC

                ground_truth = torch.arange(targets.size(-1), dtype=torch.long, device=device)  # C
                ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).expand(c_outputs.shape[0], c_outputs.shape[1],
                                                                             c_outputs.shape[2])  # 12xBxC
                regularizer_loss = torch.nn.CrossEntropyLoss(reduction='none')(scores.permute(1, 2, 3, 0), ##Modify
                                                                               ground_truth.permute(1, 2, 0))  # BxCx12
                regularizer_loss = torch.mean(
                    torch.mean(torch.sum(regularizer_loss * targets.unsqueeze(-1), dim=-2), dim=-1) / (
                                torch.sum(targets, dim=-1) + 1e-8))
                metric_logger.update(attn_loss=regularizer_loss.item())
                loss = loss + args.loss_weight*regularizer_loss  #CHange accordingly

            if patch_outputs is not None:
                ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, sal in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sal = sal.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output, cams1, cams2, proj = model(images)

            if len(output) == 2:
                output, patch_output = output
            elif len(output) == 3:
                output, c_output, patch_output = output

            loss_cls = criterion(output, target)
            batch_predict_1 = cam_labels(cams1, target, 3)  
            batch_predict_2 = cam_labels(cams2, target, 3)                   
            loss_sal = F.mse_loss(batch_predict_1, sal) + F.mse_loss(batch_predict_2, sal) #+ F.mse_loss(att2, at_gt2)
            
            loss = loss_cls + loss_sal

            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)


        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mAP=metric_logger.mAP, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True) 

    #Added lines   
    args.save_type = list() 
    if args.crf is not None:
        args.crf_list = list()
        for t in args.crf_t:
            for alpha in args.crf_alpha:
                crf_folder = os.path.join(args.crf, 'crf_{}_{}'.format(t, alpha))
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(crf_folder)

    model.eval()

    img_list = open(os.path.join(args.img_list, 'test.txt')).readlines() #Change accordinlgy
    # print(len(img_list))
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            # print(len(image_list))
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                orig_img = PIL.Image.open(os.path.join(args.data_path, 'JPEGImages', img_name + '.jpg')).convert("RGB")
                orig_img = np.asarray(orig_img)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size
                
                output, cls_attentions, patch_attn = model(images, saliency = None, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
                # print(output.shape) 
                # patch_attn = torch.sum(patch_attn, dim=0) # B*196*196 #This part only for the PASCAL VOC
                # if args.patch_attn_refine:
                #     cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0] #3*200*200
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device) #1*3*200*200

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) >= 1e-8:
                    cam_dict = {}
                    norm_cam = np.zeros((args.nb_classes, w_orig, h_orig))
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            # cls_attention = cls_attention / (np.max(cls_attention, (1, 2), keepdims=True) + 1e-5)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention
                            norm_cam[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict) 
                    if args.crf is not None:
                        for folder, t, alpha in args.crf_list:
                            cam_crf = _crf_with_alpha(orig_img, cam_dict, alpha, t=t)
                            np.save(os.path.join(folder, img_name + '.npy'), cam_crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return

def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = utils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]
    return n_crf_al

def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)