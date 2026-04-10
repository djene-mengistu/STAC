
export CUDA_VISIBLE_DEVICES=0
python ./STAC/seg/train_model_seg.py \
                    --num_epochs 50 \
                    --mask_path ./data/NEU_VOC/SegmentationMASK \
                    --data_path ./data/NEU_VOC \
                    --log_dir ./STAC/full_NEU_segmentation_output \
                    --img_list ./STAC/neu_seg \
                    --batch_size 16 \
python ./STAC/seg/infer_seg.py --weights ./STAC/full_NEU_segmentation_output/Checkpoints/dlv3_res18.pth \
                      --list_path STAC/neu_seg/test.txt \
                      --gt_path ./data/NEU_VOC/SegmentationMASK \
                      --img_path ./data/NEU_VOC/JPEGImages \
                      --save_path ./STAC/full_NEU_segmentation_dlv3_res18 \
                      --save_path_c ./STAC/full_NEU_segmentation_c_dlv3_res18 \
                      --num_classes 4 \
                      --scales 1.0 \
                      --use_crf False
# # python ./STAC/seg/train_seg.py \
# #                     --num_epochs 60 \
# #                     --network resnet38_seg \
# #                     --img_path ./data/NEU_VOC/JPEGImages \
# #                     --list_path ./STAC/neu_seg/train_val.txt \
# #                     --init_weights ./EPS_mod/outputs/cls/checkpoint_cls.pth \
# #                     --seg_pgt_path ./STAC/seg/neu_pseudo_mask \
# #                     --save_path  ./STAC/neu_segmentation_output \
# #                     --num_classes 4 \
# #                     --batch_size 16  
# python ./STAC/seg/infer_seg.py --weights ./STAC/neu_segmentation_output/model_59_final.pth \
#                       --list_path STAC/neu_seg/test.txt \
#                       --gt_path ./data/NEU_VOC/SegmentationMASK \
#                       --img_path ./data/NEU_VOC/JPEGImages \
#                       --network resnet38_seg \
#                       --save_path ./STAC/NEU_segmentation_res38 \
#                       --save_path_c ./STAC/NEU_segmentation_c_res38 \
#                       --scales 1.0 2.0\
#                       --use_crf False
# # python ./STAC/seg/train_model_seg.py \
# #                     --num_epochs 60 \
# #                     --mask_path ./STAC/seg/dagm_pseudo_mask \
# #                     --data_path ./data/nDAGM_VOC \
# #                     --log_dir ./STAC/dagm_segmentation_output \
# #                     --img_list ./STAC/dagm_seg \
# #                     --batch_size 16
# # python ./STAC/seg/infer_seg.py --weights ./STAC/dagm_segmentation_output/Checkpoints/segf0_dagm_mlp.pth \
# #                       --list_path STAC/dagm_seg/train.txt \
# #                       --gt_path ./data/nDAGM_VOC/SegmentationMASK \
# #                       --img_path ./data/nDAGM_VOC/JPEGImages \
# #                       --save_path ./STAC/dagm_segmentation_segf0_mlp \
# #                       --save_path_c ./STAC/dagm_segmentation_c_segf0_mlp \
# #                       --scales 1.0 2.0\
# #                       --use_crf False
# #JUST FOR TESTING
# # python ./STAC/seg/infer_seg.py --weights ./STAC/segmentation_output/Checkpoints/dlv3plus_effb4_raw.pth \
# #                       --list_path STAC/neu_seg/train_val.txt \
# #                       --gt_path ./data/NEU_VOC/SegmentationMASK \
# #                       --img_path ./data/NEU_VOC/JPEGImages \
# #                       --save_path ./STAC/segmentation_check1 \
# #                       --save_path_c ./STAC/segmentation_check2 \
# #                       --scales 1.0  2.0\
# #                       --use_crf False