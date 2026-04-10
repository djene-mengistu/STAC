import os

import torch
from torch.nn import functional as F

from eps import cam_labels, pico_inputs, cam_labels_pos
from pico_loss import compute_pico_loss, label_onehot #, pixel_contrastive_loss_saliency_sampled, pixel_contrastive_loss_cam_sampled
from util import pyutils
# import importlib
# from tensorboardX import SummaryWriter

def train_cls(train_loader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.max_iters):
        try:
            img_id, img, label = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            img_id, img, label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        pred = model(img)

        # Classification loss
        loss = F.multilabel_soft_margin_loss(pred, label)
        # loss = F.binary_cross_entropy_with_logits(pred, label)
        avg_meter.add({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step-1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  'Loss:%.4f' % (avg_meter.pop('loss')),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

        timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_eps(train_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_pico')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    for iteration in range(args.max_iters):
        try:
            img_id, img, saliency, label = next(loader_iter)
        except:
            loader_iter = iter(train_dataloader)
            img_id, img, saliency, label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        # print('IMG', img.shape)
        saliency = saliency.cuda(non_blocking=True)
        # print('SAL', saliency.shape)
        label = label.cuda(non_blocking=True)
        # print('LAB', label.shape)
        pred, cam = model(img)
        # print('PROJ', proj.shape)

        # print('PRED, CAM', pred.shape, cam.shape)

        # Classification loss
        # loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)
        loss_cls = F.multilabel_soft_margin_loss(pred, label)

        # loss_sal, fg_map, bg_map, sal_pred = \
        #     get_eps_loss(cam, saliency, args.num_classes, label,
        #                  args.tau, args.lam, intermediate=True)
        # loss_sal= get_eps102_loss(cam, saliency, args.lam)
        # cam_pred = cam_to_mask(cam, label)
        cam_pred = cam_labels_pos(cam, label)
        # probs, mask, label = pico_inputs(cam, saliency, label, 0.5, 3)
        loss_sal = F.mse_loss(cam_pred, saliency)
        # loss_pico = compute_pico_loss(proj, label, mask, probs, 0.8, 0.6, 256, 256)
        loss = loss_cls.float() + loss_sal.float() #+ 0.3*loss_pico.float()

        avg_meter.add({'loss': loss.item(),
                       'loss_cls': loss_cls.item(),
                       'loss_sal': loss_sal.item()})
                    #    'loss_pico': loss_pico.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step-1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                  'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                #   'Loss_Pico:%.4f' % (avg_meter.pop('loss_pico')),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

        timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_eps.pth'))