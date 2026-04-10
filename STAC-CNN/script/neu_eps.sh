#!/usr/bin/env sh
# NEED TO SET
# # 1. train classification network with EPS
# CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/main.py \
#     --dataset neu_seg \
#     --train_list /data/djene/djene/EPS_CAM/metadata/neu_seg/train.txt \
#     --session eps \
#     --network network.resnet38_eps \
#     --data_root /data/djene/djene/data/NEU_VOC/JPEGImages \
#     --saliency_root /data/djene/djene/data/NEU_VOC/SALmapsALL \
#     --weights /data/djene/djene/dseg/EPS102/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
#     --save_root /data/djene/djene/EPS_CAM/NEU_CAM_CON \
#     --lr 0.001 \
#     --wt_dec 5e-3 \
#     --batch_size 16 \
#     --crop_size 224 \
#     --max_iters 20000 \
# # # # # # 2. inference CAM
# CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/infer_CAM.py \
#     --dataset neu_seg \
#     --infer_list /data/djene/djene/EPS_CAM/metadata/neu_seg/test.txt \
#     --img_root /data/djene/djene/data/NEU_VOC/JPEGImages \
#     --tasks test \
#     --network network.resnet38_eps \
#     --weights /data/djene/djene/EPS_CAM/NEU_CAM_CON/eps/checkpoint_eps.pth \
#     --thr 0.5 \
#     --n_gpus 1 \
#     --n_processes_per_gpu 1 \
#     --cam_vis /data/djene/djene/EPS_CAM/NEU_CAM_CON/eps/cam_png \
#     --cam_npy /data/djene/djene/EPS_CAM/NEU_CAM_CON/eps/cam_npy
# # # 3. evaluate CAM
# # GT_ROOT=${DATASET_ROOT}/SegmentationMASK/
CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/eval_CAM.py \
    --datalist /data/djene/djene/EPS_CAM/metadata/neu_seg/test.txt \
    --pred_dir /data/djene/djene/EPS_CAM/NEU_CAM_CON/eps/cam_npy \
    --gt_dir /data/djene/djene/data/NEU_VOC/SegmentationMASK \
    --save_path /data/djene/djene/EPS_CAM/NEU_CAM_CON/eps/test_results.txt \
    --t 0.5 \
# # CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/eval_CAM.py \
#     --dataset neu_seg \
#     --datalist /data/djene/djene/EPS_CAM/metadata/neu_seg/test.txt \
#     --gt_dir /data/djene/djene/data/NEU_VOC/SegmentationMASK \
#     --save_path /data/djene/djene/EPS_CAM/NEU_CLS_OUT/cls/test_results.txt \
#     --pred_dir /data/djene/djene/EPS_CAM/NEU_CLS_OUT/cls/cam_npy_test
# # # # # 3. evaluate CAM
# CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_mod/eval_new.py \
#     --dataset mtd_seg \
#     --datalist /data/djene/djene/MCTCon/mtd_seg/25test.txt \
#     --gt_dir /data/djene/djene/data/MTD_VOC_25/SegmentationMASK \
#     --save_path /data/djene/djene/EPS_mod/mtd_outputs_EPS/25_test_result_eps.txt \
#     --pred_dir /data/djene/djene/EPS_mod/mtd_outputs_EPS/eps/cam_npy