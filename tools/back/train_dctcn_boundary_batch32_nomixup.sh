MODEL_JSON_PATH=configs/lrw_resnet18_dctcn_boundary.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --alpha 0 \
                                      --batch-size 32 --exp-name _batch32-lr0.0003-nomixup \
				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_dctcn_boundary_batch32-lr0.0003-nomixup/2022-09-29T21:28:29/ckpt.pth \
				      --init-epoch 1
