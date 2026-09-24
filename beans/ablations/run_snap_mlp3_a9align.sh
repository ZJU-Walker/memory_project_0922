#!/usr/bin/env bash
# A9/B9-aligned SNAP: 8 conditioned queries at layer 8, 3x1024 output-delta bank.
# Old losses, pad75, state mask .5, decay .99; RTC15; KI base -> own A500 -> B3000.
# GPUS=0,1,2,3 BATCH=16 ACCUM=1 WORKERS=8 bash beans/ablations/run_snap_mlp3_a9align.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_snap_mlp3_a9align exec bash "$D/run_stages.sh"
