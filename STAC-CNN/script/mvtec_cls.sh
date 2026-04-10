#!/usr/bin/env sh
# NEED TO SET
# 1. train classification network
# dataset=mvtec_seg
cate=leather
save_path=MVTEC_CLS_LEATHER_OUT
CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/main.py \
    --dataset mvtec_seg \
    --train_list /data/djene/djene/EPS_CAM/mvtec/$cate/all_images.txt \
    --session cls \
    --network network.resnet38_cls \
    --data_root /data/djene/djene/data/MVTec_VOC/$cate/rJPEGImages \
    --weights /data/djene/djene/dseg/EPS102/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --max_iters 3000 \
    --batch_size 16 \
    --save_root /data/djene/djene/EPS_CAM/$save_path
2. inference CAM
INFER_DATA=train # train / train_aug
TRAINED_WEIGHT= /data/djene/djene/EPS_CAM/NEU_CLS_OUT/cls/checkpoint_cls.pth
CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/infer_CAM.py \
    --dataset mvtec_seg \
    --tasks test \
    --infer_list /data/djene/djene/EPS_CAM/mvtec/$cate/def_images.txt \
    --img_root /data/djene/djene/data/MVTec_VOC/$cate/rJPEGImages \
    --network network.resnet38_cls \
    --weights /data/djene/djene/EPS_CAM/$save_path/cls/checkpoint_cls.pth \
    --thr 0.6 \
    --n_gpus 1 \
    --n_processes_per_gpu 1 \
    --cam_npy /data/djene/djene/EPS_CAM/$save_path/cam_npy \
    --cam_vis /data/djene/djene/EPS_CAM/$save_path/cam_png

# # 3. evaluate CAM
# # GT_ROOT=${DATASET_ROOT}/SegmentationMASK/
CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/eval_CAM.py \
    --datalist /data/djene/djene/EPS_CAM/mvtec/$cate/def_images.txt \
    --pred_dir /data/djene/djene/EPS_CAM/$save_path/cam_npy \
    --gt_dir /data/djene/djene/data/MVTec_VOC/$cate/rSegmentationMASK \
    --save_path /data/djene/djene/EPS_CAM/$save_path/test_results.txt \
    --t 0.5 \
# # CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/eval_CAM.py \
#     --dataset neu_seg \
#     --datalist /data/djene/djene/EPS_CAM/metadata/neu_seg/test.txt \
#     --gt_dir /data/djene/djene/data/NEU_VOC/SegmentationMASK \
#     --save_path /data/djene/djene/EPS_CAM/NEU_CLS_OUT/cls/test_results.txt \
#     --pred_dir /data/djene/djene/EPS_CAM/NEU_CLS_OUT/cls/cam_npy_test