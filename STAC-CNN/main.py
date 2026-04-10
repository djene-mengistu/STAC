import os
import torch
import argparse
from torch.backends import cudnn

from util import pyutils

from module.dataloader import get_dataloader
from module.model import get_model
# from more_models.mit_eps import get_model
from module.optimizer import get_optimizer
from module.train import train_cls, train_eps
from thop import profile
cudnn.enabled = True
torch.backends.cudnn.benchmark = False

_NUM_CLASSES = {'dagm_seg': 6, 'neu_seg': 3, 'mtd_seg': 5, 'mvtec_seg': 1}

def get_arguments():
    parser = argparse.ArgumentParser()
    # session
    parser.add_argument("--session", default="eps", type=str)

    # data
    parser.add_argument("--data_root", default="/home/dengjie/djene/data/MTD_VOC/JPEGImages", type=str)
    parser.add_argument("--dataset", default='mtd_seg', type=str)
    parser.add_argument("--saliency_root", default="/home/dengjie/djene/data/MTD_VOC/SALmapsALL",type=str)
    parser.add_argument("--train_list", default="/home/dengjie/djene/EPS_mod/metadata/mtd_seg/train.txt", type=str)
    parser.add_argument("--save_root", default='/home/dengjie/djene/EPS_mod/mtd_outputs')

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--resize_size", default=(256, 256), type=int, nargs='*')

    # network
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", default='/data/djene/djene/dseg/EPS102/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)

    # optimizer
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-6, type=float)
    parser.add_argument("--max_iters", default=4000, type=int)

    # hyper-parameters for EPS
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--lam", default=0.5, type=float)

    args = parser.parse_args()

    args.num_classes = _NUM_CLASSES[args.dataset]

    if 'cls' in args.network:
        args.network_type = 'cls'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    else:
        raise Exception('No appropriate model type')

    return args


if __name__ == '__main__':

    # get arguments
    args = get_arguments()

    # set log
    args.log_folder = os.path.join(args.save_root, args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    pyutils.Logger(os.path.join(args.log_folder, 'mtd_log_cls.log'))
    print(vars(args))

    # load dataset
    train_loader = get_dataloader(args)

    max_step = args.max_iters

    # load network and its pre-trained model
    model = get_model(args)
    # print(model)
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {num_params}")

    # MACs and FLOPs
    # import time
    # input_image = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model, inputs=(input_image,))
    # print(f"MACs: {macs / 1e9} GMacs")
    # print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs
    # batch_size = 1
    # input_shape = (3, 224, 224)  # Change if needed
    # dummy_input = torch.randn(batch_size, *input_shape).cuda()

    # # Warm-up (important for accurate timing on GPU)
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = model(dummy_input)

    # # Timed run
    # n_iterations = 1000  # Number of inference passes
    # start_time = time.time()

    # with torch.no_grad():
    #     for _ in range(n_iterations):
    #         _ = model(dummy_input)

    # end_time = time.time()
    # total_time = end_time - start_time
    # avg_time_per_image = total_time / (n_iterations * batch_size)
    # fps = 1.0 / avg_time_per_image

    # # Get peak memory usage in MB
    # # peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # # print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
    # memory_usage = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    # peak_memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # Peak memory in MB
    # print(f"Current GPU Memory Usage: {memory_usage:.2f} MB")
    # print(f"Peak GPU Memory Usage: {peak_memory_usage:.2f} MB")

    # print(f"Average inference time: {avg_time_per_image * 1000:.2f} ms")
    # print(f"FPS: {fps:.2f}")
    # model = get_model(4)

    # set optimizer
    optimizer = get_optimizer(args, model, max_step)

    # train
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    if args.network_type == 'cls':
        train_cls(train_loader, model, optimizer, max_step, args)
    elif args.network_type == 'eps':
        train_eps(train_loader, model, optimizer, max_step, args)
    else:
        raise Exception('No appropriate model type')