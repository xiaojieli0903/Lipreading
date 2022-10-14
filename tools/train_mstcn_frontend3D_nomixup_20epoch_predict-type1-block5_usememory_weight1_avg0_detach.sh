MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data

GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py --modality video --config-path $MODEL_JSON_PATH --annonation-direc $ANNONATION_DIRECTORY --data-dir $MOUTH_ROIS_DIRECTORY \
    --alpha 0 --epochs 20  --predict-loss-weight 1 --loss-average-dim 0 \
    --exp-name _nomixup_20epochs_lw1_avg0_detach --detach-target

#CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
#                                      --config-path $MODEL_JSON_PATH \
#                                      --annonation-direc $ANNONATION_DIRECTORY \
#                                      --data-dir $MOUTH_ROIS_DIRECTORY \
#				      --alpha 0 --epochs 20  --predict-loss-weight 1 --loss-average-dim 0 \
#				      --exp-name _nomixup_20epochs_lw1_avg0_detach --detach-target
