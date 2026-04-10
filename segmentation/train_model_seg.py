import argparse
import os
# import sys
# sys.path.append('../')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import importlib
from pathlib import Path
from datetime import datetime
from distutils.dir_util import copy_tree
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
import argparse

# import torch.backends.cudnn as cudnn
# import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from dataloader_seg import build_dataset
# from network.custom_models.Transformer_based import Transformer_model
from network.custom_models.deit_mix import DEITMIX
 
from tool.metrics_iou import*
from tool.dice_loss import DiceLoss
from tool.utilities import get_logger, create_dir
from segmentation_models_pytorch import DeepLabV3Plus #Install the modules of segmentation model


parser = argparse.ArgumentParser()
parser.add_argument('--nb_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--log_dir', type=str,  default='/home/dengjie/djene/MCTCon/segmentation_output', help='mask path')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_epochs', type=int,  default=60, help='number of epochs')
parser.add_argument('--bach_size', type=int,  default=16, help='batch size')
parser.add_argument('--img_list', type=str,  default='/home/dengjie/djene/MCTCon/dagm_seg', help='image list')
parser.add_argument('--data_path', type=str,  default='/home/dengjie/djene/data/NEU_VOC', help='image path')
parser.add_argument('--mask_path', type=str,  default='/home/dengjie/djene/MCTCon/seg/pseudo_gt_raw', help='mask path')
# parser.add_argument('--mask_path', type=str,  default='/home/dengjie/djene/data/NEU_VOC/SegmentationMASK', help='mask path')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU id's, GPU id's start from 0.

#python setting
# seed = 1337
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Define the segmentation models
# model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)
# NET = Transformer_model('MiT-B4', 6) #
# NET.init_pretrained('/data/djene/djene/MCTCon/seg/network/custom_models/weights/mit_b4.pth')
# model = NET
# model = DeepLabV3Plus("efficientnet-b4", encoder_weights="imagenet", classes=4, activation=None) #Change accordingly
# model = DeepLabV3Plus("resnet18", encoder_weights="imagenet", classes=4, activation=None) #Change accordingly
model = DEITMIX(num_classes=4, pretrained=True)
dice_loss = DiceLoss()
base_lr = args.base_lr

dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
dataset_test, _ = build_dataset(is_train=False, args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_test = torch.utils.data.SequentialSampler(dataset_test)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=12,
    pin_memory=True,
    drop_last=True, #Change accordingly
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    sampler=sampler_test,
    batch_size=args.batch_size,
    num_workers=12,
    pin_memory=True,
    drop_last=False, #Change accordingly
)

class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model = model
        self._init_logger()
    def _init_logger(self):

        log_dir = args.log_dir

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        self.model.to(device)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.000001, patience=8, verbose=True)      
        
        self.logger.info(
            "train_loader {} test_loader {}".format(len(data_loader_train), len(data_loader_test)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(0, args.num_epochs):

            running_ce_loss = 0.0
            running_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_iou_1 = 0.0
            running_train_dice_1 = 0.0
            
            running_val_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_ce_loss = 0.0
                        
            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            running_val_accuracy_1 = 0.0
            
            optimizer_1.zero_grad()
            
            self.model.train()
                    
            # with torch.cuda.amp.autocast():
                
            for iteration, data in enumerate (data_loader_train): #(zip(train_loader, unlabeled_train_loader)):
                
                inputs_S1, labels_S1 = data                
                
                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)                
                
                self.model.train()

                # Train Model 1
                outputs_1 = self.model(inputs_S1)
                outputs_soft_1 = torch.softmax(outputs_1, dim=1)
                
                loss_ce_1 = ce_loss(outputs_1, labels_S1.long())
                # loss_dice_1 = dice_loss(labels_S1.unsqueeze(1), outputs_1)
                loss_dice_1 = dice_loss(outputs_1, labels_S1)
                
                loss = 0.5 * (loss_dice_1 + loss_ce_1)                                      
                
                optimizer_1.zero_grad()
                
                loss.backward()
                optimizer_1.step()
                running_train_loss += loss.item()
                running_ce_loss += loss_ce_1.item()
                running_dice_loss += loss_dice_1.item()
                
                running_train_iou_1 += mIoU(outputs_1, labels_S1)
                running_train_dice_1 += mDice(outputs_1, labels_S1)
                
                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']
                
                iter_num = iter_num + 1
            
            epoch_loss = (running_train_loss) / (len(data_loader_train))
            epoch_ce_loss = (running_ce_loss) / (len(data_loader_train))
            epoch_dice_loss = (running_dice_loss) / (len(data_loader_train))
            epoch_train_iou = (running_train_iou_1) / (len(data_loader_train))
            epoch_train_dice = (running_train_dice_1) / (len(data_loader_train))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                            format(datetime.now(), epoch, args.num_epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train dice: {}'.format(epoch_train_dice))
            self.writer.add_scalar('Train/mDice', epoch_train_dice, epoch)
            self.logger.info('Train IoU: {}'.format(epoch_train_iou))
            self.writer.add_scalar('Train/mIoU', epoch_train_iou, epoch)

            self.writer.add_scalar('info/lr', lr_, epoch)
            torch.cuda.empty_cache()

            self.model.eval() 
            for i, pack in enumerate(data_loader_test, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images = images.to(device)
                    gts = gts.to(device)                    
                    prediction_1 = self.model(images)
                    
                loss_ce_1 = ce_loss(prediction_1, gts.long())
                loss_dice_1 = 1 - mDice(prediction_1, gts)
                val_loss = 0.5 * (loss_dice_1 + loss_ce_1)
                running_val_loss += val_loss
                running_val_dice_loss += loss_dice_1
                running_val_ce_loss += loss_ce_1
                
                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)
                
            epoch_loss_val = running_val_loss / len(data_loader_test)
            epoch_ce_loss_val = running_val_ce_loss / len(data_loader_test)
            epoch_dice_loss_val = running_val_dice_loss / len(data_loader_test)
            epoch_dice_val_1 = running_val_dice_1 / len(data_loader_test)
            epoch_iou_val_1 = running_val_iou_1 / len(data_loader_test)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(data_loader_test)
            scheduler_1.step(epoch_dice_val_1)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)

            self.logger.info('Val CE-loss: {}'.format(epoch_ce_loss_val))
            self.writer.add_scalar('Val/CE-loss', epoch_ce_loss_val, epoch)

            self.logger.info('Val Dice-loss: {}'.format(epoch_dice_loss_val))
            self.writer.add_scalar('Val/Dice-loss', epoch_dice_loss_val, epoch)

            #model-1 perfromance
            self.logger.info('Val dice : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Val/DSC', epoch_dice_val_1, epoch)

            self.logger.info('Val IoU : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Val/IoU', epoch_iou_val_1, epoch)

            self.logger.info('Val Accuracy : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Val/Accuracy', epoch_accuracy_val_1, epoch)
            
            self.writer.add_scalar('info/lr', lr_, epoch)
            # self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            torch.cuda.empty_cache()            
            
            mdice_coeff_1 =  epoch_dice_val_1

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1
                        
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                # torch.save(state_1, Checkpoints_Path + '/segformer_mit_b0_mlp_raw.pth')
                torch.save(state_1, Checkpoints_Path + '/dlv3_res18.pth')    
            
            
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))

            self.logger.info('current patience :{}'.format(self.patience))
            # print('Current consistency weight:', consistency_weight)
            print('================================================================================================')
            print('================================================================================================')




if __name__ == '__main__':
    train_network = Network()
    train_network.run()