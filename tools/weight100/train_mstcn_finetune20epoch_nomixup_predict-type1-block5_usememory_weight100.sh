MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --model-path ./train_logs/tcn/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory_finetune20epochs_batch64_lr3e-5_nomixup_lw100/2022-10-10T10:56:13/ckpt.pth \
				      --alpha 0 --allow-size-mismatch --predict-loss-weight 100 \
				      --init-epoch 1 --epochs 20 --batch-size 64 --lr 3e-5 --exp-name _finetune20epochs_batch64_lr3e-5_nomixup_lw100
