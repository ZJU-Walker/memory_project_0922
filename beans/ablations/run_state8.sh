#!/usr/bin/env bash
# Row state8: snap + state-only sensory bank, delta rule; 8 sentence + 8 sensory memory tokens, 4 cards.
#   [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_state8.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_state8 EXP=${EXP:-ab_state8} exec bash "$D/train_ablation.sh"
