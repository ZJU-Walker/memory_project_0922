#!/usr/bin/env bash
# Ablation (1) "snap + visual memory" (8 sentence + 8 visual memory tokens) on 4 cards.
#   [JOB=<slurm job>] [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_vis8.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_vis8 EXP=${EXP:-ab_vis8} exec bash "$D/train_ablation.sh"
