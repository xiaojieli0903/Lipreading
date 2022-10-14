MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend2D_predict-future_usememory.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=2 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 \
				      --exp-name _nomixup \
				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn_frontend2D_predict-future_usememory_nomixup/2022-10-05T18:41:14/ckpt.pth \
				      --init-epoch 1
