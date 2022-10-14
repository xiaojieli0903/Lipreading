MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend2D.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --model-path train_logs/tcn/lrw_resnet18_mstcn_frontend2D/2022-10-03T17:08:08/ckpt.pth \
				      --init-epoch 1
