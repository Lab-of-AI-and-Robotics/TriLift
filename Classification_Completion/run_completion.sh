#!/usr/bin/env bash
# TriLift — Completion (our paper method)
# Default run = TriLift-F with transformer positional encoding.
#
# Usage:
#   bash run_completion.sh                  # ratio 0.5  -> TriLift-F(1/2)  [default]
#   bash run_completion.sh 0.25             # ratio 0.25 -> TriLift-F(1/4)
#   bash run_completion.sh 0.5 --epochs 100 # extra args are passed through
#
# The first positional argument is the low-resolution branch ratio (--3dratio).
# If omitted it defaults to 0.5.
set -e

RATIO="${1:-0.5}"
if [ $# -gt 0 ]; then shift; fi

python completion.py --pe_type transformer --3dratio "${RATIO}" "$@"
