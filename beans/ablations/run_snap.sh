#!/usr/bin/env bash
# Control: automatically derived sentence slots; own A250 -> B3000.
#   [JOB=<slurm job>] [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_snap.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_snap exec bash "$D/run_stages.sh"
