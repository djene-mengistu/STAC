from torch.backends import cudnn

cudnn.enabled = True
from tool import pyutils, torchutils
import argparse
import importlib
import tool.exutils as exutils

import torch.nn.functional as F
from pathlib import Path
from thop import profile
import torch
import os
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU_id')

    parser.add_argument("--list_path", default="home/dengjie/djene/MCTCon/neu_seg/train_val.txt", type=str)
    parser.add_argument("--img_path", default="/home/dengjie/djene/data/NEU_VOC/JPEGImages", type=str)
    parser.add_argument("--save_path", default="/home/dengjie/djene/MCTCon/segmentation_output", type=str)
    parser.add_argument("--seg_pgt_path", default="/home/dengjie/djene/MCTCon/seg/pseudo_gt", type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--num_epochs", default=60, type=int)
    parser.add_argument("--network", default='resnet38_seg', type=str)
    parser.add_argument("--lr", default=0.0008, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--init_weights", default='/home/dengjie/djene/EPS_mod/outputs/cls/checkpoint_cls.pth', type=str)

    parser.add_argument("--session_name", default="model_", type=str)
    parser.add_argument("--crop_size", default=200, type=int)

    parser.add_argument('--print_intervals', type=int, default=50)

    args = parser.parse_args()

    gpu_id = args.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    save_path = args.save_path
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    pyutils.Logger(os.path.join(args.save_path, args.session_name + '.log'))

    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()

    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes).cuda()

    weights_dict = torch.load(args.init_weights)
    model.load_state_dict(weights_dict, strict=False)
    # model.eval()
    # # print(model)
    # # num_params = sum(p.numel() for p in model.parameters())
    # # print(f"Number of parameters: {num_params}")

    # # # MACs and FLOPs
    # # input_image = torch.randn(1, 3, 224, 224)
    # # macs, params = profile(model, inputs=(input_image,))
    # # print(f"MACs: {macs / 1e9} GMacs")
    # # print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs
    # import time
    # batch_size = 1
    # input_shape = (3, 224, 224)  # Change if needed
    # dummy_input = torch.randn(batch_size, *input_shape).cuda()

    # # # Warm-up (important for accurate timing on GPU)
    # # with torch.no_grad():
    # #     for _ in range(10):
    # #         _ = model(dummy_input)

    # # # Timed run
    # # n_iterations = 1000  # Number of inference passes
    # # start_time = time.time()

    # # with torch.no_grad():
    # #     for _ in range(n_iterations):
    # #         _ = model(dummy_input)

    # # end_time = time.time()
    # # total_time = end_time - start_time
    # # avg_time_per_image = total_time / (n_iterations * batch_size)
    # # fps = 1.0 / avg_time_per_image
    # # Warm-up
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = model(dummy_input)
    
    # # Timed run
    # torch.cuda.synchronize()  # Ensure GPU is ready
    # start_time = time.time()
    
    # with torch.no_grad():
    #     for _ in range(1000):
    #         _ = model(dummy_input)
    
    # torch.cuda.synchronize()  # Wait for GPU to finish
    # end_time = time.time()
    
    # # Calculate FPS
    # total_time = end_time - start_time
    # avg_time_per_image = total_time / (1000 * batch_size)
    # fps = 1.0 / avg_time_per_image
    
    # # Print results
    # print(f"Total Inference Time: {total_time:.2f} seconds")
    # print(f"Average Time per Image: {avg_time_per_image:.4f} seconds")
    # print(f"FPS: {fps:.2f}")

    # # Get peak memory usage in MB
    # # peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # # print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
    # memory_usage = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    # peak_memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # Peak memory in MB
    # print(f"Current GPU Memory Usage: {memory_usage:.2f} MB")
    # print(f"Peak GPU Memory Usage: {peak_memory_usage:.2f} MB")

    # # print(f"Average inference time: {avg_time_per_image * 1000:.2f} ms")
    # # print(f"FPS: {fps:.2f}")

    img_list = exutils.read_file(args.list_path)
    train_size = len(img_list)
    num_batches_per_epoch = train_size // args.batch_size
    max_step = args.num_epochs * num_batches_per_epoch

    data_list = []
    for i in range(200):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    optimizer = torchutils.PolyOptimizer_cls([
        {'params': model.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.get_10x_lr_params(), 'lr': 10 * args.lr}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    data_gen = exutils.chunker(data_list, args.batch_size)

    for ep in range(args.num_epochs):
        for iter in range(num_batches_per_epoch):
            chunk = data_gen.__next__()
            img_list = chunk

            images, ori_images, seg_labels, img_names = exutils.get_data_from_chunk(chunk, args)

            b, _, w, h = ori_images.shape
            seg_labels = seg_labels.long().cuda()
            images = images.cuda()

            pred = model(x=images)
            pred = F.interpolate(pred, size=(w, h), mode='bilinear', align_corners=False)
            loss = criterion(pred, seg_labels)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_intervals == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.5f' % (optimizer.param_groups[0]['lr']), flush=True)
            if ep % 10 ==0:
                torch.save(model.module.state_dict(), os.path.join(save_path, args.session_name + str(ep) + '.pth'))

    torch.save(model.module.state_dict(), os.path.join(save_path, args.session_name + str(ep) + '_final'+'.pth'))