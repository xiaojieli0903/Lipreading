MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 --allow-size-mismatch --predict-loss-weight 1 --loss-average-dim 0 \
				      --init-epoch 0 --epochs 20 --batch-size 64 --lr 3e-5 --exp-name _finetune20epochs_batch64_lr3e-5_nomixup_lw1_avg0 \
				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn/2022-09-24T15:18:32/ckpt.best.pth
