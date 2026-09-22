#!/usr/bin/env bash
# Bean scoop with the 0920 v1 structure on the 2 x H100 (job $JOB, default 17489557): ONE self-write run from the boba base 9999
# (config pi05_yam_beans_0920_v1; a fresh experiment dir, run_train.sh starts fresh). Run ON iris-hgx-1 via train_beans_v1_ctl.sh.
# The 1 GB train_hs keep-alives are never touched; the run refuses to start while any process holds > 2 GB on the cards.
# MODE=smoke -> pi05_yam_beans_0920_v1_smoke (3 updates). BATCH_FALLBACK="2": on RESOURCE_EXHAUSTED retry once per listed batch.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92}
JOB=${JOB:-17489557}; BATCH=${BATCH:-4}; WORKERS=${WORKERS:-12}; MODE=${MODE:-train}; REMAT=${REMAT:-nothing_saveable}
BATCH_FALLBACK=${BATCH_FALLBACK:-2}
case $MODE in
  smoke) CFG=pi05_yam_beans_0920_v1_smoke; EXP=${EXP:-smoke_beans_0920_v1}; WANDB=0 ;;
  *)     CFG=pi05_yam_beans_0920_v1; EXP=${EXP:-beans_0920_v1}; WANDB=${WANDB:-1} ;;
esac
status=$ROOT/beans/logs/train_beans_v1_status.log
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
gpu_busy() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; }
wait_gpu() { for i in $(seq 1 240); do [ "$(gpu_busy)" = "0" ] && return 0; sleep 30; done; return 1; }
log "start mode=$MODE host=$(hostname) job=$JOB batch=$BATCH fallback='$BATCH_FALLBACK' workers=$WORKERS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION remat=$REMAT cfg=$CFG exp=$EXP code=$(git -C $ROOT rev-parse --short HEAD)"
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
run_once() { OPENPI_BEANS_BATCH=$1 OPENPI_0920_REMAT=$REMAT LOGS=$ROOT/beans/logs CKBASE=$ROOT/beans/checkpoints WANDB=$WANDB JOB=$JOB GRES=2 VISIBLE=0,1 GPUS=2 BATCH=$1 ACCUM=1 CPUS=16 bash cluster_robomme/run_train.sh "$CFG" "$EXP" --num-workers $WORKERS; }
for b in $BATCH $BATCH_FALLBACK; do
  log "attempt batch=$b remat=$REMAT exp=$EXP"
  run_once $b
  st=$(tail -1 "$ROOT/beans/logs/train_${EXP}_status.log")
  if echo "$st" | grep -q "^exit=0"; then log "end mode=$MODE batch=$b: $st"; exit 0; fi
  if tail -400 "$ROOT/beans/logs/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
    log "batch $b ran out of memory ($st)"; rm -rf "$ROOT/beans/checkpoints/$CFG/$EXP"; sleep 30; wait_gpu || { log "cards busy after OOM"; exit 1; }; continue
  fi
  log "end mode=$MODE batch=$b (not an OOM): $st"; exit 1
done
log "end mode=$MODE: every batch ran out of memory"; exit 1
