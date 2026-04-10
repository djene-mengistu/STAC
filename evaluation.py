import os
import pandas as pd
import numpy as np
from PIL import Image
import dill
import multiprocessing
import argparse
import torchvision.transforms.functional as F

categories = ['background','inclusion','patches','scratches'] #NEU-Seg
# categories = ['background','class1','class2','class6', 'class7','class9','class10'] #DAGM
# categories = ['background','Blowhole','Break','Crack', 'Fray','Uneven'] #MTD

def compare(start,step,TP,P,T,input_type,threshold,predict_folder, gt_folder, sal_folder, name_list, num_cls=4):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_cls,h,w),np.float32)
                
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold #1.0 - saliency_map 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png'%name) #Change accordingly NEU-Seg and MTD, .png, else .PNG
            gt_mask = Image.open(gt_file)
            # # gt_mask = gt_mask.resize((224,224), resample=0) #Changed line
            # # gt_mask = F.resize(gt_mask, (224, 224), interpolation=F.InterpolationMode.NEAREST)
            # gt_mask = gt_mask.resize((224, 224), Image.NEAREST)
            gt = np.array(gt_mask)
            # gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt)* cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()

def do_python_eval(predict_folder, gt_folder, sal_folder, name_list, num_cls=4, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    
    p_list = []
    # p = compare(i,8,TP,P,T,input_type,threshold)
    # p_list.append(p)
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold, args.predict_dir, args.gt_dir, args.sal_dir, name_list, args.num_classes))
        p.start()
        p_list.append(p)
    for p in p_list: 
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            # print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='C:/Users/djmen/Desktop/MCT+/neu_seg/test.txt', type=str)
    parser.add_argument("--predict_dir", default='C:/Users/djmen/Desktop/MCT+/new_pred_dir/cam-npy', type=str)
    parser.add_argument("--gt_dir", default='D:/DAGM_VOC/SegmentationMASK', type=str)
    parser.add_argument("--sal_dir", default='D:/DAGM_VOC/SALmapsALL', type=str)
    parser.add_argument('--logfile', default='C:/Users/djmen/Desktop/MCT+/new_pred_dir/evallog_test_840.txt',type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=0.7, type=float)
    parser.add_argument('--curve', default=True, type=bool)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--start', default=50, type=int)
    parser.add_argument('--end', default=100, type=int)
    args = parser.parse_args()

    # multiprocessing.set_start_method("spawn")

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'], dtype=str) #Modified for dtype to keep starting zeros
    name_list = df['filename'].values
    
    loglist = do_python_eval(args.predict_dir, args.gt_dir, args.sal_dir, name_list, args.num_classes, args.type, args.t, printlog=True)
    writelog(args.logfile, loglist, args.comment)
    # if not args.curve:
    #     loglist = do_python_eval(args.predict_dir, args.gt_dir, args.sal_dir, name_list, args.num_classes, args.type, args.t, printlog=True)
    #     writelog(args.logfile, loglist, args.comment)
    # else:
    #     l = []
    #     max_mIoU = 0.0
    #     best_thr = 0.0
    #     for i in range(args.start, args.end, 5):
    #         t = i/100.0
    #         loglist = do_python_eval(args.predict_dir, args.gt_dir, args.sal_dir, name_list, args.num_classes, args.type, t, printlog=True)
    #         l.append(loglist['mIoU'])
    #         print('%d/100 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
    #         if loglist['mIoU'] > max_mIoU:
    #             max_mIoU = loglist['mIoU']
    #             best_thr = t
    #         # else:
    #         #     break
    #     print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
    #     writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)

