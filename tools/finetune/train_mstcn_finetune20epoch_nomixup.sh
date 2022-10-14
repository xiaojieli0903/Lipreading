MODEL_JSON_PATH=configs/lrw_resnet18_mstcn.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcnfinetune20epochs_batch64_lr3e-5_nomixup/2022-10-09T00:00:53/ckpt.pth \
				      --alpha 0 \
				      --init-epoch 1 --epochs 20 --batch-size 64 --lr 3e-5 --exp-name finetune20epochs_batch64_lr3e-5_nomixup
