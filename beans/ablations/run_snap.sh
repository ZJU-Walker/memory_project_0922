#!/usr/bin/env bash
# The control row: snap (8 sentence memory tokens) under the same 4-card ablation recipe.
#   [JOB=<slurm job>] [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_snap.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_snap EXP=${EXP:-ab_snap} exec bash "$D/train_ablation.sh"
