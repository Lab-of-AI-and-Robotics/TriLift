set -x
set -e

DATA_ROOT=./data/front3d/front3d_rpn_data


python3 -u run_rpn.py \
--mode "eval" \
--dataset_name front3d \
--resolution 160 \
--backbone_type vgg_EF \
--features_path ./cube_results/front3d \
--boxes_path ${DATA_ROOT}/obb \
--dataset_split ${DATA_ROOT}/3dfront_split.npz \
--save_path ./results/front3d_anchor_vgg_EF \
--filter_threshold 0.5 \
--rpn_nms_thresh 0.3 \
--normalize_density \
--rotated_bbox \
--batch_size 4 \
--gpus 0 

