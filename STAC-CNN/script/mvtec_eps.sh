#!/usr/bin/env sh
# NEED TO SET
# # 1. train classification network with EPS
cate=carpet
save_path=MVTEC_CAM_CARPET_OUT
# CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/main.py \
#     --dataset mvtec_seg \
#     --train_list /data/djene/djene/EPS_CAM/mvtec/$cate/all_images.txt \
#     --session eps \
#     --network network.resnet38_eps \
#     --data_root /data/djene/djene/data/MVTec_VOC/$cate/rJPEGImages \
#     --saliency_root /data/djene/djene/data/MVTec_VOC/$cate/rSALmapsALL \
#     --weights /data/djene/djene/dseg/EPS102/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
#     --save_root /data/djene/djene/EPS_CAM/$save_path \
#     --lr 0.001 \
#     --wt_dec 5e-3 \
#     --batch_size 16 \
#     --crop_size 224 \
#     --max_iters 3000 \
# # # # # 2. inference CAM
# CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/infer_CAM.py \
#     --dataset mvtec_seg \
#     --infer_list /data/djene/djene/EPS_CAM/mvtec/$cate/def_images.txt \
#     --img_root /data/djene/djene/data/MVTec_VOC/$cate/rJPEGImages \
#     --tasks test \
#     --network network.resnet38_eps \
#     --weights /data/djene/djene/EPS_CAM/$save_path/eps/checkpoint_eps.pth \
#     --thr 0.6 \
#     --n_gpus 1 \
#     --n_processes_per_gpu 1 \
#     --cam_vis /data/djene/djene/EPS_CAM/$save_path/cam_png \
#     --cam_npy /data/djene/djene/EPS_CAM/$save_path/cam_npy
# # # 3. evaluate CAM
# # GT_ROOT=${DATASET_ROOT}/SegmentationMASK/
CUDA_VISIBLE_DEVICES=0 python /data/djene/djene/EPS_CAM/eval_CAM.py \
    --datalist /data/djene/djene/EPS_CAM/mvtec/$cate/def_images.txt \
    --pred_dir /data/djene/djene/EPS_CAM/$save_path/cam_npy \
    --gt_dir /data/djene/djene/data/MVTec_VOC/$cate/rSegmentationMASK \
    --save_path /data/djene/djene/EPS_CAM/$save_path/test_results.txt \
    --t 0.6 \