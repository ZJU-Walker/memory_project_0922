#!/usr/bin/env bash
# SNAP-MLP3: 3x1024 hidden layers, output-only delta writes, mean confidence >=0.9.
# Sampling: full/slice/transition = 25/25/50, pad 50 frames; RTC 15; own A250 -> B3000.
#   GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 bash beans/ablations/run_snap_mlp3.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_snap_mlp3 exec bash "$D/run_stages.sh"
