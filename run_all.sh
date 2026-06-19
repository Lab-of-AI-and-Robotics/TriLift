#!/usr/bin/env bash
# ============================================================================
# run_all.sh  —  Full TriLift pipeline (ours only).
# For each task:  data download  ->  [preprocessing]  ->  train at ratio 0.5
# (TriLift-F 1/2) and 0.25 (TriLift-F 1/4).  Original epochs: cls/comp 50,
# segmentation 100, detection 60.  Detection runs a train pass then an eval
# pass on the final checkpoint (per-epoch eval is disabled in run_rpn.py),
# producing Table IV metrics (Recall@25/50, AP@25/50).
# Order: classification, completion, segmentation, detection.
# Usage (activate the conda env first, then run from anywhere):
#   conda activate trilift
#   bash run_all.sh 2>&1 | tee run_all.log
# ============================================================================
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Repo root = the directory containing this script (no hardcoded absolute paths).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "#### RUN ALL START $(date) ####"

run() {
  local name="$1"; shift
  local dir="$1"; shift
  echo ""
  echo ">>>>> [$name] START $(date) <<<<<"
  local t0=$SECONDS
  ( cd "$ROOT/$dir" && "$@" )
  local ec=$?
  local dt=$(( SECONDS - t0 ))
  if [ $ec -eq 0 ]; then echo ">>>>> [$name] PASS (exit 0, ${dt}s) <<<<<"
  else echo ">>>>> [$name] FAIL (exit $ec, ${dt}s) <<<<<"; fi
}

# detection: train (60 ep) then eval the final checkpoint -> Table IV metrics.
# det <dataset> <resolution> <batch> <splitfile> <ratio>
det() {
  local ds="$1" res="$2" bs="$3" split="$4" r="$5"
  run "det_${ds}_F${r}_train" Object_Detection python3 -u run_rpn.py \
    --mode train --dataset_name "$ds" --resolution "$res" --backbone_type vgg_EF \
    --pe_type transformer --3dratio "$r" \
    --features_path ./cube_results/"$ds" \
    --boxes_path ./data/"$ds"/"${ds}_rpn_data"/obb \
    --dataset_split ./data/"$ds"/"${ds}_rpn_data"/"$split" \
    --save_path ./results/"${ds}_F${r}" \
    --num_epochs 60 --lr 3e-4 --weight_decay 1e-3 \
    --log_interval 20 --eval_interval 1 --rpn_nms_thresh 0.3 \
    --log_to_file --normalize_density --rotated_bbox --batch_size "$bs" --gpus 0
  local edir="results/eval_${ds}_F${r}"
  ( cd "$ROOT/Object_Detection"
    last=$(ls results/"${ds}_F${r}"/epoch_*.pt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)\.pt/\1/' | sort -n | tail -1)
    rm -rf "$edir"; mkdir -p "$edir"
    [ -n "$last" ] && ln -sf "../${ds}_F${r}/epoch_${last}.pt" "$edir/epoch_${last}.pt" )
  run "det_${ds}_F${r}_eval" Object_Detection python3 -u run_rpn.py \
    --mode eval --dataset_name "$ds" --resolution "$res" --backbone_type vgg_EF \
    --pe_type transformer --3dratio "$r" \
    --features_path ./cube_results/"$ds" \
    --boxes_path ./data/"$ds"/"${ds}_rpn_data"/obb \
    --dataset_split ./data/"$ds"/"${ds}_rpn_data"/"$split" \
    --save_path "./$edir" \
    --filter_threshold 0.5 --rpn_nms_thresh 0.3 \
    --normalize_density --rotated_bbox --batch_size "$bs" --gpus 0
}

# ============================================================================
# 1. Classification  (ModelNet40)
# ============================================================================
# Data download (Google Drive)
( cd "$ROOT/Classification_Completion/datasets" \
  && gdown 1s_lM8yAaQ8xXEpau6kTbMlH4Inng2sXI -O modelnet40_ply_hdf5_2048.tar.gz \
  && tar xzf modelnet40_ply_hdf5_2048.tar.gz )
# Preprocessing (point cloud -> binary occupancy voxels)
( cd "$ROOT/Classification_Completion" && python prepare_classification.py )
# Train (TriLift-F)
run "classification_F0.5"  Classification_Completion bash run_classification.sh 0.5
run "classification_F0.25" Classification_Completion bash run_classification.sh 0.25

# ============================================================================
# 2. Completion  (ModelNet40)
# ============================================================================
# Data download (Google Drive)
( cd "$ROOT/Classification_Completion/datasets" \
  && gdown 1sShW_ItA7yX0_8yBvAH70FzOEBKnvOS9 -O ModelNet40.tar.gz \
  && tar xzf ModelNet40.tar.gz )
# Preprocessing (mesh -> solid occupancy voxels)
( cd "$ROOT/Classification_Completion" && python prepare_completion.py --save )
# Train (TriLift-F)
run "completion_F0.5"  Classification_Completion bash run_completion.sh 0.5
run "completion_F0.25" Classification_Completion bash run_completion.sh 0.25

# ============================================================================
# 3. Semantic Segmentation  (ScanNet / Stanford3D)
# ============================================================================
# Data download: ScanNet & Stanford3D are license-gated; download from their
# official sources, generate the preprocessed/ folder via
# SpatioTemporalSegmentation (branch v0.5), then preprocess:
#   python prepare_segmentation.py --save
# Train (TriLift-F)
for R in 0.5 0.25; do
  run "segmentation_scannet_F${R}"    Semantic_Segmentation bash run_segmentation.sh "$R" --dataset scannet
  run "segmentation_stanford3d_F${R}" Semantic_Segmentation bash run_segmentation.sh "$R" --dataset stanford3d
done

# ============================================================================
# 4. Object Detection  (NeRF-RPN: Front3D / Hypersim / ScanNet)
# ============================================================================
# Data download: NeRF-RPN dataset from HuggingFace into Object_Detection/data,
# then preprocess (rgbsigma .npz -> cube features):
#   python data_modify.py --dataset front3d   # and hypersim, scannet
# Train + eval (TriLift-F)
for R in 0.5 0.25; do
  det front3d  160 4 3dfront_split.npz  "$R"
  det hypersim 200 2 hypersim_split.npz "$R"
  det scannet  160 4 scannet_split.npz  "$R"
done

echo ""
echo "#### RUN ALL DONE $(date) ####"
