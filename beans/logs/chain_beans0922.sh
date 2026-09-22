#!/usr/bin/env bash
# The beans0922 chain: base to 10k, then the memory run from base/10000 to 5k. Same knobs as train_beans0922.sh (JOB, GPUS, ...).
# Skips the base when its 10000 checkpoint already exists. Run detached (beans0922_ctl.sh start).
set -u
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
status="$ROOT/beans/logs/train_beans0922_status.log"; log() { echo "[$(date +%m/%d\ %H:%M:%S)] chain: $*" | tee -a "$status"; }
base_ckpt="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/${BASE_EXP:-beans0922_base}/10000/params"
if [ -d "$base_ckpt" ]; then log "base 10000 present, skipping the base run"; else
  log "stage 1: base"; MODE=base EXP=${BASE_EXP:-beans0922_base} bash "$ROOT/beans/logs/train_beans0922.sh" || { log "base failed; chain stopped"; exit 1; }
  [ -d "$base_ckpt" ] || { log "base finished without $base_ckpt; chain stopped"; exit 1; }
fi
log "stage 2: memory run from $base_ckpt"; MODE=mem EXP=${MEM_EXP:-beans0922_v1} bash "$ROOT/beans/logs/train_beans0922.sh"; log "chain done exit=$?"
