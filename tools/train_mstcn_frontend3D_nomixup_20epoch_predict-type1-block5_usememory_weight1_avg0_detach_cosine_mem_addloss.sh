MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory_mvm.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=2 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 --epochs 20  --predict-loss-weight 1 --loss-average-dim 0 \
				      --exp-name _nomixup_20epochs_lw1_avg0_cosine_addloss --detach-target \
				      --predict-loss-type 'cosine' --add-memory-loss --contrastive-loss-weight 0
