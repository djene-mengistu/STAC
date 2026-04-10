import torch
from torch.nn import functional as F
from pico_loss import label_onehot #, pixel_contrastive_loss_saliency_sampled, pixel_contrastive_loss_cam_sampled

def cam_to_mask(cams, targets, target_size=(224, 224)):
    """
    Generate a differentiable, soft pseudo-label mask from CAMs using image-level labels.
    
    Args:
        cams: (B, C, H, W) raw CAMs (logits or unnormalized)
        targets: (B, C) binary multi-label targets (0 or 1)
        target_size: (H, W) to interpolate to
    
    Returns:
        mask: (B, 1, H, W) soft attention mask in [0, 1]
    """
    B, C, H, W = cams.shape
    device = cams.device

    # 1. Interpolate to target size
    cams_up = F.interpolate(cams, size=target_size, mode='bilinear', align_corners=False)  # (B, C, H, W)

    # 2. Zero out CAMs for negative classes (using targets as mask)
    # targets: (B, C) → expand to (B, C, 1, 1)
    targets_mask = targets.view(B, C, 1, 1)
    cams_pos = cams_up * targets_mask  # (B, C, H, W); negative classes = 0

    # 3. Optional: Apply ReLU to keep only positive activations
    cams_pos = F.relu(cams_pos)

    # 4. Normalize per-sample to [0, 1] (differentiable)
    # Compute min and max per sample (over spatial and class dims for positive classes only)
    # But simpler: normalize per-sample over all spatial locations
    cams_flat = cams_pos.view(B, -1)  # (B, C*H*W)
    cams_min = cams_flat.min(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
    cams_max = cams_flat.max(dim=1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

    # Avoid division by zero
    eps = 1e-8
    cams_norm = (cams_pos - cams_min) / (cams_max - cams_min + eps)  # (B, C, H, W)

    # 5. Combine: sum over classes (since saliency is class-agnostic)
    mask = cams_norm.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    # 6. Optional: clamp to [0,1] (already should be, but safe)
    mask = torch.clamp(mask, 0, 1)

    return mask

def cam_labels(cams, targets, num_classes=3):
    cam = F.relu(cams)
    cls_attentions = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    # cls_attentions = F.relu(cls_attentions)
    # at_gt = cls_attentions* (targets.view(cls_attentions.shape[0], 3, 1, 1).expand(cls_attentions.shape[0], 3, 224, 224))

    batch_predict = []
    for b in range(cams.shape[0]):
        if (targets[b].sum()) > 1e-5: #Check the effect of this value, without the use of the >>if 
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
            # max_values, _ = torch.max(tensor, dim=0) 
            predict = torch.sum(tensor, dim=0)  
            # Normalize to [0, 1]
            # predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)         
            # predict = (max_values >= 0.4).byte()                   
            batch_predict.append(predict)

    batch_predict = torch.stack(batch_predict, dim=0)
    batch_predict = batch_predict.unsqueeze(1)
    return batch_predict

def cam_labels_pos(cams, targets, output_size=(224, 224)):
    """
    Generate binary attention maps from 1-channel CAMs.
    Output shape: (B, 1, H, W)
    """
    cam = F.relu(cams)
    cls_attentions = F.interpolate(cam, size=output_size, mode='bilinear', align_corners=False)
    batch_predict = []
    for b in range(cams.shape[0]):
        if targets[b] > 1e-5:
            cam_fg = cls_attentions[b, 0]
            cam_fg = (cam_fg - cam_fg.min()) / (cam_fg.max() - cam_fg.min() + 1e-8)
            batch_predict.append(cam_fg)
        else:
            batch_predict.append(torch.zeros_like(cls_attentions[b, 0]))
    return torch.stack(batch_predict, dim=0).unsqueeze(1)

def pico_inputs(cams, sal, targets, weak_threshold, num_classes):
    bcg = 1 - sal #Compute the saleincy map
    cam = F.relu(cams)
    cls_attentions = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False) 
    # print('CLS', cls_attentions.shape)

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

def get_eps_loss(cam, saliency, num_classes, label, tau, lam, intermediate=True):
    """
    Get EPS loss for pseudo-pixel supervision from saliency map.
    Args:
        cam (tensor): response from model with float values.
        saliency (tensor): saliency map from off-the-shelf saliency model.
        num_classes (int): the number of classes
        label (tensor): label information.
        tau (float): threshold for confidence area
        lam (float): blending ratio between foreground map and background map
        intermediate (bool): if True return all the intermediates, if not return only loss.
    Shape:
        cam (N, C+1, H', W') where N is the batch size and C is the number of classes.
        saliency (N, 1, H, W)
        label (N, C)
    """
    b, c, h, w = cam.size()
    saliency = F.interpolate(saliency, size=(h, w))

    label_map = label.view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w)).bool()

    # Map selection
    label_map_fg = torch.zeros(size=(b, num_classes + 1, h, w)).bool().cuda()
    label_map_bg = torch.zeros(size=(b, num_classes + 1, h, w)).bool().cuda()

    label_map_bg[:, num_classes] = True
    label_map_fg[:, :-1] = label_map.clone()

    sal_pred = F.softmax(cam, dim=1)

    iou_saliency = (torch.round(sal_pred[:, :-1].detach()) * torch.round(saliency)).view(b, num_classes, -1).sum(-1) / \
                   (torch.round(sal_pred[:, :-1].detach()) + 1e-04).view(b, num_classes, -1).sum(-1)

    valid_channel = (iou_saliency > tau).view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w))

    label_fg_valid = label_map & valid_channel

    label_map_fg[:, :-1] = label_fg_valid
    label_map_bg[:, :-1] = label_map & (~valid_channel)

    # Saliency loss
    fg_map = torch.zeros_like(sal_pred).cuda()
    bg_map = torch.zeros_like(sal_pred).cuda()

    fg_map[label_map_fg] = sal_pred[label_map_fg]
    bg_map[label_map_bg] = sal_pred[label_map_bg]

    fg_map = torch.sum(fg_map, dim=1, keepdim=True)
    bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)
    sal_pred = fg_map * lam + bg_map * (1 - lam)

    loss = F.mse_loss(sal_pred, saliency)

    if intermediate:
        return loss, fg_map, bg_map, sal_pred
    else:
        return loss

def get_eps102_loss(cam, saliency):
    """
    Get EPS loss for pseudo-pixel supervision from saliency map with improved gradient flow and supervision.
    Args:
        cam (tensor): response from model with float values.
        saliency (tensor): saliency map from off-the-shelf saliency model.
        num_classes (int): the number of classes
        label (tensor): label information.
        tau (float): threshold for confidence area
        lam (float): blending ratio between foreground map and background map
        alpha (float): weight for foreground loss
        beta (float): weight for background loss
        intermediate (bool): if True return all the intermediates, if not return only loss.
    Shape:
        cam (N, C+1, H', W') where N is the batch size and C is the number of classes.
        saliency (N, 1, H, W)
        label (N, C)
    """
    b, c, h, w = cam.size()
    
    # 1️⃣ Interpolate the saliency map smoothly to match the CAM size
    saliency = F.interpolate(saliency, size=(h, w), mode='bilinear', align_corners=True)
    
    # print(saliency.shape)
    
    # 4️⃣ Use softmax to get predicted class probabilities
    sal_pred = F.softmax(cam, dim=1)  # (b, c+1, h, w)
    # sal_pred = sal_pred.sum(dim=1, keepdim=True)  # (b, 1, h, w)
    fg_map = sal_pred[:, 1:, :, :].sum(dim=1, keepdim=True)  # (b, 1, h, w)
    bg_map = sal_pred[:, 0, :, :].unsqueeze(1) # (b, 1, h, w)
    # print(bg_map.shape)
    # fg_map = torch.sum(fg_map, dim=1, keepdim=True)
    # bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)#.unsqueeze(1)
    # print(bg_map.shape)
    sal_pred = fg_map + bg_map
    
    loss = F.mse_loss(sal_pred, saliency)  
    return loss

def get_sal_loss(cam, saliency):
    """
    SAL LOSS
    """
    b, c, h, w = cam.size() 
    loss = F.mse_loss(cam, saliency)  
    return loss
