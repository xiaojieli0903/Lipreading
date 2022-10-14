MODEL_JSON_PATH=$1
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
MODEL_PATH=$2
CUDA_VISIBLE_DEVICES=1 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --model-path $MODEL_PATH \
                                      --extract-feats --mouth-patch-path /disk/gao2/datasets/lrw/test_list.txt --mouth-embedding-out-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn/2022-09-24T15:18:32/features
