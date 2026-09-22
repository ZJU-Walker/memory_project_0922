#!/usr/bin/env bash
# Row state8_add: snap + state-only sensory bank, ADDITIVE rule; 8 sentence + 8 sensory memory tokens, 4 cards.
#   [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_state8_add.sh [smoke]
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_state8_add EXP=${EXP:-ab_state8_add} exec bash "$D/train_ablation.sh"
