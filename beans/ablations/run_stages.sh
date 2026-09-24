#!/usr/bin/env bash
# Same run_<row>.sh entrypoints; each row owns its A -> B chain.
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
ROW=${CFG#pi05_yam_beans0922_ab_}
case "$ROW" in snap|vis8|vis8s|state8|vis8s_add|state8_add) ;; *) exit 2 ;; esac
MODE=${MODE:-train}; case "$MODE" in train|smoke) ;; *) exit 2 ;; esac
RUN_NAME=${RUN_NAME:-slot_${ROW}}
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
suffix=; a_steps=${A_STEPS:-250}; b_steps=${STEPS:-3000}
if [ "$MODE" = smoke ]; then suffix=_smoke; RUN_NAME="smoke_${RUN_NAME}"; a_steps=2; b_steps=2; fi
LOGS="$ROOT/beans/ablations/logs"; mkdir -p "$LOGS"
exec 9>"$LOGS/${RUN_NAME}.lock"
flock -n 9 || { echo "$RUN_NAME already has a launcher"; exit 2; }
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_BEANS_AB_A_STEPS="$a_steps" OPENPI_BEANS_AB_STEPS="$b_steps"
export OPENPI_BEANS_AB_BATCH=${BATCH:-16} OPENPI_BEANS_AB_WORKERS=${WORKERS:-8}
export OPENPI_BEANS_AB_ACCUM=${ACCUM:-1}
export OPENPI_BEANS_BASE_PARAMS=${OPENPI_BEANS_BASE_PARAMS:-$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params}
export OPENPI_BEANS_AB_A_PARAMS="$ROOT/beans/checkpoints/pi05_yam_beans0922_ab_${ROW}_A${suffix}/${RUN_NAME}_A/${a_steps}/params"
[ -d "$OPENPI_BEANS_BASE_PARAMS" ] || { echo "Missing KI base/10000: $OPENPI_BEANS_BASE_PARAMS"; exit 2; }
for stage in A B; do
  config="pi05_yam_beans0922_ab_${ROW}"; steps=$b_steps
  if [ "$stage" = A ]; then config="${config}_A"; steps=$a_steps; fi
  config="${config}${suffix}"; exp="${RUN_NAME}_${stage}"
  ckpt="$ROOT/beans/checkpoints/$config/$exp"; marker="$LOGS/${exp}.recipe"
  signature="template_slot_ab_v1 row=$ROW stage=$stage batch=$OPENPI_BEANS_AB_BATCH accum=$OPENPI_BEANS_AB_ACCUM steps=$steps A=$a_steps prefill=${OPENPI_BEANS_AB_PREFILL_STEPS:-320} base=$OPENPI_BEANS_BASE_PARAMS"
  if [ -f "$marker" ]; then
    [ "$(< "$marker")" = "$signature" ] || { echo "Recipe changed: choose a new RUN_NAME"; exit 2; }
  elif [ -e "$ckpt" ]; then
    echo "Refusing unmarked checkpoint $ckpt; choose a new RUN_NAME"; exit 2
  fi
  printf '%s\n' "$signature" > "$marker"
  if [ -d "$ckpt/$steps/params" ]; then echo "$stage already completed: $ckpt/$steps"; continue; fi
  [ "$stage" = A ] || [ -d "$OPENPI_BEANS_AB_A_PARAMS" ] || { echo 'Missing own A checkpoint'; exit 2; }
  echo "$(date -Is) $ROW $stage: $steps updates, batch=$OPENPI_BEANS_AB_BATCH"
  CFG="$config" EXP="$exp" MODE="$MODE" bash "$ROOT/beans/ablations/train_slot_stage.sh"
done
