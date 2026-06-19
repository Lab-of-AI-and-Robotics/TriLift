#!/usr/bin/env bash
# TriLift — Semantic Segmentation (our paper method)
# Default run = TriLift-F with transformer positional encoding (ScanNet).
#
# Usage:
#   bash run_segmentation.sh                          # ratio 0.5, ScanNet  [default]
#   bash run_segmentation.sh 0.25                      # ratio 0.25 -> TriLift-F(1/4)
#   bash run_segmentation.sh 0.5 --dataset stanford3d  # Stanford3D
#   bash run_segmentation.sh 0.5 --epochs 100          # extra args are passed through
#
# The first positional argument is the low-resolution branch ratio (--3dratio).
# If omitted it defaults to 0.5.
set -e

RATIO="${1:-0.5}"
if [ $# -gt 0 ]; then shift; fi

python segmentation.py --pe_type transformer --3dratio "${RATIO}" "$@"
