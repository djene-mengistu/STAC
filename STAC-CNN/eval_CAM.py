import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time

# UPDATE THIS TO MATCH YOUR DATASET
# For MTD: background + 5 defect classes → 6 total
# categories = ['background', 'inclusion', 'patches', 'scratches']
categories = ['background', 'Bad']
# If you only have 4 defect types, set num_classes=5 and use:
# categories = ['background', 'inclusion', 'patches', 'scratches', 'class3']

def compare(start, step, TP, P, T, input_type, threshold, predict_folder, gt_folder, name_list, num_cls):
    for idx in range(start, len(name_list), step):
        name = name_list[idx]
        if input_type == 'png':
            predict_file = os.path.join(predict_folder, f'{name}.png')
            predict = np.array(Image.open(predict_file))
            if num_cls == 81:  # for COCO/Pascal
                predict = predict - 91
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder, f'{name}.npy')
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            # Create score tensor: [bg, cls0, cls1, ..., cls_{C-1}]
            tensor = np.full((num_cls, h, w), -np.inf, dtype=np.float32)
            tensor[0, :, :] = threshold  # background score
            for cls_idx in predict_dict:
                # cls_idx is 0-based foreground index (0 to num_cls-2)
                if cls_idx < num_cls - 1:
                    tensor[cls_idx + 1, :, :] = predict_dict[cls_idx]
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
        else:
            raise ValueError("input_type must be 'png' or 'npy'")

        gt_file = os.path.join(gt_folder, f'{name}.png')
        gt = np.array(Image.open(gt_file))
        valid_mask = gt < 255  # ignore void/ignore labels

        for i in range(num_cls):
            pred_i = (predict == i) & valid_mask
            gt_i = (gt == i) & valid_mask
            tp_i = (pred_i & gt_i)

            TP[i].acquire()
            TP[i].value += np.sum(tp_i)
            TP[i].release()

            P[i].acquire()
            P[i].value += np.sum(pred_i)
            P[i].release()

            T[i].acquire()
            T[i].value += np.sum(gt_i)
            T[i].release()


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=2, input_type='npy', threshold=0.5, printlog=False, num_workers=8):
    TP = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    P = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    T = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]

    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=compare,
            args=(i, num_workers, TP, P, T, input_type, threshold, predict_folder, gt_folder, name_list, num_cls)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    IoU = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        tp = TP[i].value
        p = P[i].value
        t = T[i].value
        union = t + p - tp + 1e-10
        iou = tp / union
        fp = (p - tp) / union
        fn = (t - tp) / union
        IoU.append(iou)
        FP_ALL.append(fp)
        FN_ALL.append(fn)

    loglist = {}
    for i in range(num_cls):
        if i < len(categories):
            loglist[categories[i]] = IoU[i] * 100
        else:
            loglist[f'class_{i}'] = IoU[i] * 100

    miou = np.mean(IoU)
    loglist['mIoU'] = miou * 100
    loglist['FP'] = np.mean(FP_ALL) * 100
    loglist['FN'] = np.mean(FN_ALL) * 100

    if printlog:
        print("\nClass-wise IoU:")
        for i in range(num_cls):
            cat = categories[i] if i < len(categories) else f'cls{i}'
            print(f'{cat:>12}: {IoU[i]*100:6.2f}%')
        print('======================================================')
        print(f'{"mIoU":>12}: {miou*100:6.2f}%')
        print(f'{"FP":>12}: {loglist["FP"]:6.2f}%')
        print(f'{"FN":>12}: {loglist["FN"]:6.2f}%\n')

    return loglist


def writelog(filepath, metric, comment):
    with open(filepath, 'a') as logfile:
        logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logfile.write(f'\t{comment}\n')
        for key, value in metric.items():
            if isinstance(value, (list, np.ndarray)):
                logfile.write(f'{key}: {value}\n')
            else:
                logfile.write(f'{key}: {value:.4f}\n')
        logfile.write('=====================================\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", default='/data/djene/djene/EPS_mod/metadata/mtd_seg/test.txt', type=str)
    parser.add_argument("--pred_dir", default='/data/djene/djene/EPS_mod/mtd_outputs_EPS/eps/cam_npy', type=str)
    parser.add_argument("--gt_dir", default='/data/djene/djene/data/MTD_VOC/SegmentationMASK', type=str)
    parser.add_argument("--save_path", default='/data/djene/djene/EPS_mod/mtd_outputs_EPS/mtd_result_eps.txt', type=str)
    parser.add_argument("--comment", default="mtd-388", type=str)
    parser.add_argument("--type", default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument("--t", default=0.5, type=float)
    parser.add_argument("--curve", action='store_true')
    parser.add_argument("--num_classes", default=2, type=int)  # 5 defects + background
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    args = parser.parse_args()

    # Validate categories length
    if len(categories) != args.num_classes:
        print(f"Warning: categories has {len(categories)} names, but num_classes={args.num_classes}")
        # Extend if needed
        while len(categories) < args.num_classes:
            categories.append(f'class_{len(categories)}')

    df = pd.read_csv(args.datalist, names=['filename'], dtype=str)
    name_list = df['filename'].values

    if not args.curve:
        loglist = do_python_eval(
            args.pred_dir, args.gt_dir, name_list,
            num_cls=args.num_classes,
            input_type=args.type,
            threshold=args.t,
            printlog=True,
            num_workers=args.num_workers
        )
        writelog(args.save_path, loglist, args.comment)
    else:
        best_miou = 0.0
        best_thr = 0.0
        miou_list = []
        print(f"Threshold sweep: {args.start}% to {args.end}%")
        for i in range(args.start, args.end + 1, 3):
            thr = i / 100.0
            loglist = do_python_eval(
                args.pred_dir, args.gt_dir, name_list,
                num_cls=args.num_classes,
                input_type=args.type,
                threshold=thr,
                printlog=False,
                num_workers=args.num_workers
            )
            miou_list.append(loglist['mIoU'])
            if loglist['mIoU'] > best_miou:
                best_miou = loglist['mIoU']
                best_thr = thr
            print(f'Threshold: {thr:.2f} | mIoU: {loglist["mIoU"]:.2f}%')

        print(f'\nBest: threshold={best_thr:.2f}, mIoU={best_miou:.2f}%')
        writelog(args.save_path, {
            'mIoU_curve': miou_list,
            'Best mIoU': best_miou,
            'Best threshold': best_thr
        }, args.comment)