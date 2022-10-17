MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block7_usememory_mvm_fixmemory_bycontext.json
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 --epochs 20  --predict-loss-weight 10 --loss-average-dim 0 \
				      --exp-name _nomixup_20epochs_lw10_cosine_recon10_contrastive1 --detach-target \
				      --predict-loss-type 'cosine' --add-memory-loss --recon-loss-weight 10 --contrastive-loss-weigh 1

