#!/usr/bin/env bash
# Row vis8s_add: snap + vision + state sensory bank, ADDITIVE rule (repeats accumulate); 8 sentence + 8 sensory memory tokens, 4 cards.
#   [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_vis8s_add.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_vis8s_add EXP=${EXP:-ab_vis8s_add} exec bash "$D/train_ablation.sh"
