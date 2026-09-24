#!/usr/bin/env bash
# SNAP + visual: induced sentence slots + 8 auxiliary read tokens. H100: BATCH=12 ACCUM=1.
#   [JOB=<slurm job>] [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_vis8.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_vis8 exec bash "$D/run_stages.sh"
