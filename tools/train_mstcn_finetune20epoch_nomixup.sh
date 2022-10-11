MODEL_JSON_PATH=configs/lrw_resnet18_mstcn.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn/2022-09-24T15:18:32/ckpt.best.pth \
				      --alpha 0 \
				      --init-epoch 0 --epochs 20 --batch-size 64 --lr 3e-5 --exp-name finetune20epochs_batch64_lr3e-5_nomixup