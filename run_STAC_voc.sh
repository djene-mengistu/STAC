# ####### test MCTformerPlus ##########
export CUDA_VISIBLE_DEVICES=1
# python ./STAC/main.py  --data-path ./data/VOC2012 \
#                 --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
#                 --output_dir ./STAC/EXP_VOC/STAC-STAC/saved_model \
#                 --img-list ./STAC/voc12 \
#                 --data-set VOC12 \
#                 --temp 0.3 \
#                 --strong_thresh 1.0 \
#                 --num_queries 256 \
#                 --num_negatives 512
# # # # # # # ## Generating class-specific localization maps ##########
# python ./STAC/main.py --data-set VOC12MS \
#                 --img-list ./STAC/voc12 \
#                 --data-path ./data/VOC2012 \
#                 --gen_attention_maps \
#                 --resume ./STAC/EXP_VOC/STAC-STAC/saved_model/checkpoint.pth \
#                 --cam-npy-dir ./STAC/EXP_VOC/STAC-STAC/cam-npy-fused-valval \
#                 --attention-dir ./STAC/EXP_VOC/STAC-STAC/cam-png-fused-valval \
#                 --attention-type fused\
#                 --layer-index 12 \
#                 --saliency None
# # # #### Evaluating the generated class-specific localization maps ##########
python ./STAC/evaluation.py --list ./STAC/voc12/val_id.txt \
                --gt_dir ./data/VOC2012/SegmentationClassAug \
                --sal_dir ./data/VOC2012/SaliencyMaps \
                --logfile ./STAC/EXP_VOC/STAC-STAC/evallog_valval_fused_0.66.txt \
                --type npy \
                --curve True \
                --t 0.66 \
                --predict_dir ./STAC/EXP_VOC/STAC-STAC/cam-npy-fused-valval \
                --comment "val_id"
######### Evaluating the generated class-specific localization maps ##########
# python ./STAC/evaluation_voc12.py --list ./STAC/voc12/val_id.txt \
#                      --gt_dir ./data/VOC2012/SegmentationClassAug \
#                      --logfile ./STAC/EXP_VOC/32-p-dim/evallog.txt \
#                      --type npy \
#                      --curve True \
#                      --predict_dir ./STAC/EXP_VOC/32-p-dim/cam-npy-patchcam-val \
#                      --comment "val1464"
# python ./STAC/main.py  --data-path ./data/MTD_VOC \
#                 --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
#                 --output_dir ./STAC/mtd_out/saved_model \
#                 --img-list ./STAC/mtd_seg \
#                 --data-set neu_seg \
#                 --temp 0.5 \
#                 --strong_thresh 1.0 \
#                 --num_queries 256 \
#                 --num_negatives 256
# # # # # # # # # # # ## Generating class-specific localization maps ##########
# python ./STAC/main.py --data-set neu_seg_MS \
#                 --img-list ./STAC/mtd_seg \
#                 --data-path ./data/MTD_VOC \
#                 --gen_attention_maps \
#                 --resume ./STAC/mtd_out/saved_model/checkpoint.pth \
#                 --cam-npy-dir ./STAC/mtd_out/cam-npy-fused-test \
#                 --attention-dir ./STAC/mtd_out/cam-png-fused-test \
#                 --attention-type fused \
#                 --layer-index 12 \
#                 --saliency None
# # #### Evaluating the generated class-specific localization maps ##########
# python ./STAC/evaluation.py --list ./STAC/mtd_seg/new_test.txt \
#                 --gt_dir ./data/MTD_VOC/SegmentationMASK \
#                 --sal_dir ./data/MTD_VOC/SALmapsALL \
#                 --logfile ./STAC/mtd_out/evallog_fused_fused_0.4.txt \
#                 --type npy \
#                 --curve True \
#                 --t 0.5 \
#                 --predict_dir ./STAC/mtd_out/cam-npy-fused-test \
#                 --comment "test"