#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=./data/front3d/front3d_rpn_data

python3 -u run_rpn.py \
--mode train \
--dataset_name front3d \
--resolution 160 \
--backbone_type vgg_EF \
--features_path ./cube_results/front3d \
--boxes_path ${DATA_ROOT}/obb \
--dataset_split ${DATA_ROOT}/3dfront_split.npz \
--save_path ./results/front3d_anchor_vgg_EF \
--num_epochs 60 \
--lr 3e-4 \
--weight_decay 1e-3 \
--log_interval 20 \
--eval_interval 1 \
--rpn_nms_thresh 0.3 \
--log_to_file \
--normalize_density \
--rotated_bbox \
--batch_size 4 \
--gpus 0
