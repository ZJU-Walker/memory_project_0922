#!/usr/bin/env bash
# SNAP + state: induced sentence slots + 8 auxiliary reads. H100: BATCH=8 ACCUM=1.
#   [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_state8.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_state8 exec bash "$D/run_stages.sh"
