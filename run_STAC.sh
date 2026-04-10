# ####### test STAC ##########
export CUDA_VISIBLE_DEVICES=0
python ./STAC/main.py  --data-path F:/NEU_VOC/ \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                --output_dir ./STAC/best_weight/saved_model \
                --img-list ./STAC/neu_seg \
                --data-set neu_seg \
                --temp 0.6 \
                --strong_thresh 1.0 \
                --num_queries 256 \
                --num_negatives 256
# # # # # ## Generating class-specific localization maps ##########
python ./STAC/main.py --data-set neu_seg_MS \
                --img-list ./STAC/neu_seg \
                --data-path F:/NEU_VOC/ \
                --gen_attention_maps \
                --resume ./STAC/best_weight/saved_model_neu/checkpoint.pth \
                --scales 1.0 2.0 \
                --cam-npy-dir ./STAC/EXP_NEU/npy-patchcam \
                --attention-dir ./STAC/EXP_NEU/png-patchcam \
                --attention-type patchcam \
                --layer-index 12 \
                --saliency None
# # # #### Evaluating the generated class-specific localization maps ##########
python ./STAC/evaluation.py --list ./STAC/neu_seg/test.txt \
                --gt_dir F:/NEU_VOC/SegmentationClass/ \
                --sal_dir F:/NEU_VOC/SALmapsALL/ \
                --logfile ./STAC/EXP_NEU/evallog_patchcam_0.6.txt \
                --type npy \
                --curve True \
                --t 0.6 \
                --predict_dir ./STAC/EXP_NEU/npy-patchcam \
                --comment "test840"
######### Evaluating the generated class-specific localization maps ##########
# python ./evaluation_voc12.py --list ./voc12/val_id.txt \
#                      --gt_dir /data/djene/djene/data/VOC2012/SegmentationClassAug \
#                      --logfile ./EXP_VOC/32-p-dim/evallog.txt \
#                      --type npy \
#                      --curve True \
#                      --predict_dir ./EXP_VOC/32-p-dim/cam-npy-patchcam-val \
#                      --comment "val1464"
# python ./main.py  --data-path /data/djene/djene/data/MTD_VOC \
#                 --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
#                 --output_dir ./mtd_out/saved_model \
#                 --img-list ./mtd_seg \
#                 --data-set neu_seg \
#                 --temp 0.5 \
#                 --strong_thresh 1.0 \
#                 --num_queries 256 \
#                 --num_negatives 256
# # # # # # # # # # # ## Generating class-specific localization maps ##########
# python ./main.py --data-set neu_seg_MS \
#                 --img-list ./mtd_seg \
#                 --data-path /data/djene/djene/data/MTD_VOC \
#                 --gen_attention_maps \
#                 --resume ./mtd_out/saved_model/checkpoint.pth \
#                 --cam-npy-dir ./mtd_out/cam-npy-fused-test \
#                 --attention-dir ./mtd_out/cam-png-fused-test \
#                 --attention-type fused \
#                 --layer-index 12 \
#                 --saliency None
# # #### Evaluating the generated class-specific localization maps ##########
# python ./evaluation.py --list ./mtd_seg/new_test.txt \
#                 --gt_dir /data/djene/djene/data/MTD_VOC/SegmentationMASK \
#                 --sal_dir /data/djene/djene/data/MTD_VOC/SALmapsALL \
#                 --logfile ./mtd_out/evallog_fused_fused_0.4.txt \
#                 --type npy \
#                 --curve True \
#                 --t 0.5 \
#                 --predict_dir ./mtd_out/cam-npy-fused-test \
#                 --comment "test"
# python ./main.py  --data-path /data/djene/djene/data/MVTec_VOC/leather \
#                 --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
#                 --output_dir ./EXP_MVTec/withSAL202/leather/saved_model \
#                 --img-list ./mvtec/leather \
#                 --data-set mvtec_seg \
#                 --temp 0.6 \
#                 --strong_thresh 1.0 \
#                 --num_queries 256 \
#                 --num_negatives 256
# # # # ## Generating class-specific localization maps ##########
# python ./main.py --data-set mvtec_seg_MS \
#                 --img-list ./mvtec/leather \
#                 --data-path /data/djene/djene/data/MVTec_VOC/leather \
#                 --gen_attention_maps \
#                 --resume ./EXP_MVTec/withSAL202/leather/saved_model/checkpoint.pth \
#                 --cam-npy-dir ./EXP_MVTec/withSAL202/leather/npy-patchcam \
#                 --attention-dir ./EXP_MVTec/withSAL202/leather/png-patchcam \
#                 --attention-type patchcam \
#                 --layer-index 12 \
#                 --saliency None
# # # # # #### Evaluating the generated class-specific localization maps ##########
# python ./evaluation_mvtec.py --list ./mvtec/leather/def_images.txt \
#                 --gt_dir /data/djene/djene/data/MVTec_VOC/leather/rSegmentationMASK \
#                 --sal_dir /data/djene/djene/data/MVTec_VOC/leather/rSALmapsALL \
#                 --logfile ./EXP_MVTec/withSAL202/leather/evallog_patchcam_0.4.txt \
#                 --type npy \
#                 --curve True \
#                 --t 0.6 \
#                 --predict_dir ./EXP_MVTec/withSAL202/leather/npy-patchcam \
#                 --comment "mvtec"