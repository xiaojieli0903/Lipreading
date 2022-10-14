MODEL_JSON_PATH=configs/lrw_resnet18_dctcn_boundary_cbmargin.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
CUDA_VISIBLE_DEVICES=2 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --alpha 0 \
				                              --batch-size 128 --exp-name _batch128-lr0.0003-nomixup \
		#		                              --model-path /home/gao2/Lipreading_using_Temporal_Convolutional_Networks/train_logs/tcn/lrw_resnet18_dctcn_boundary_cbmargin_batch128-lr0.0003-nomixup/2022-09-29T14:28:09/ckpt.pth \
		#		                              --init-epoch 1

