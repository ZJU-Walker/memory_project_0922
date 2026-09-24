#!/usr/bin/env bash
# Opt-in B-only continuation. Never runs A or changes the standard A500 -> B3000 chain.
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
ROW=state8_mlp3_a9align
MODE=${1:-train}
case "$MODE" in train|smoke|check) ;; *) echo "usage: $0 [train|smoke|check]"; exit 2 ;; esac

# Dedicated variables avoid inheriting RUN_NAME/ACCUM/A_STEPS from the old A launcher.
RUN_NAME=${B_RUN_NAME:-slot_state8_mlp3_a9align_A250_b12_acc1}
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid B_RUN_NAME'; exit 2; }
SOURCE=${A250_PARAMS:-$ROOT/beans/checkpoints/pi05_yam_beans0922_ab_state8_mlp3_a9align_A/slot_state8_mlp3_a9align_b12_acc3_A/250/params}
[[ "$SOURCE" = /* ]] || { echo 'A250_PARAMS must be an absolute path'; exit 2; }
[ -d "$SOURCE" ] || { echo "Missing saved A250 params: $SOURCE"; exit 2; }
SOURCE="$(cd -- "$SOURCE" && pwd -P)"
case "$SOURCE" in
  */pi05_yam_beans0922_ab_state8_mlp3_a9align_A/*/250/params) ;;
  *) echo 'A250_PARAMS must point to a finalized state8_mlp3_a9align A/250/params directory'; exit 2 ;;
esac
CHECKPOINT_ROOT=${OPENPI_BEANS_AB_CHECKPOINT_ROOT:-$ROOT/beans/checkpoints}
[[ "$CHECKPOINT_ROOT" = /* ]] || { echo 'Checkpoint root must be an absolute path'; exit 2; }

steps=3000
CFG=pi05_yam_beans0922_ab_state8_mlp3_a9align
if [ "$MODE" = smoke ]; then
  steps=2; CFG="${CFG}_smoke"; RUN_NAME="smoke_${RUN_NAME}"
fi
EXP="${RUN_NAME}_B"
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_BEANS_AB_A_PARAMS="$SOURCE" OPENPI_BEANS_AB_A_STEPS=250 OPENPI_BEANS_AB_STEPS="$steps"
export OPENPI_BEANS_AB_BATCH=12 OPENPI_BEANS_AB_ACCUM=1 OPENPI_BEANS_AB_WORKERS=${WORKERS:-4}
export OPENPI_BEANS_AB_PREFILL_STEPS=${OPENPI_BEANS_AB_PREFILL_STEPS:-320}
export CFG EXP MODE RUN_NAME
echo "B-only: $CFG exp=$EXP batch=12 accum=1 target=$steps; full A250 params=$SOURCE; fresh optimizer on first launch"
if [ "$MODE" = check ]; then exit 0; fi

LOGS="$ROOT/beans/ablations/logs"
mkdir -p "$LOGS"
exec 9>"$LOGS/${RUN_NAME}.lock"
flock -n 9 || { echo "$RUN_NAME already has a launcher"; exit 2; }
ckpt="$CHECKPOINT_ROOT/$CFG/$EXP"
marker="$LOGS/${EXP}.recipe"
signature="a9align_state8_A250_B_b12_acc1_v1 row=$ROW stage=B batch=12 accum=1 steps=$steps A=250 prefill=$OPENPI_BEANS_AB_PREFILL_STEPS source=$SOURCE checkpoint_root=$CHECKPOINT_ROOT"
if [ -f "$marker" ]; then
  [ "$(< "$marker")" = "$signature" ] || { echo 'Recipe changed: choose a new B_RUN_NAME'; exit 2; }
elif [ -e "$ckpt" ]; then
  echo "Refusing unmarked checkpoint $ckpt; choose a new B_RUN_NAME"; exit 2
fi
printf '%s\n' "$signature" > "$marker"
if [ -d "$ckpt/$steps/params" ]; then echo "B already completed: $ckpt/$steps"; exit 0; fi
# Existing stage helper handles B-only resume, GPU-idle checks and 1GB keep-alives.
# The B config restores ALL source A parameters; no A optimizer state is imported.
exec bash "$ROOT/beans/ablations/train_slot_stage.sh"
