#!/usr/bin/env bash
# 0920_v1 on the 4 x H200 (job $JOB; user 09-21 15:08: "action horizon 30 ... use 4h200 ... bigger batch ... make sure this v1
# start from plain base pi05"): ONE self-write run from the ORIGINAL pi05 base (config pi05_robomme_0920_v1 -> loader =
# openpi-assets pi05_base, memory leaves fresh; a new experiment dir, so run_train.sh starts fresh and never resumes a v0 dir).
# Run ON the node via train_0920_v1_ctl.sh. The 1 GB train_hs keep-alive is never touched.
# MODE=smoke -> pi05_robomme_0920_v1_smoke (3 updates); MODE=probe -> pi05_robomme_0920_v1_probe (31 updates, one 40-tick shape).
# BATCH_FALLBACK="40 32": if the run dies with RESOURCE_EXHAUSTED, retry once per listed batch (fresh dir each time).
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
JOB=${JOB:-17422727}; BATCH=${BATCH:-32}; WORKERS=${WORKERS:-24}; MODE=${MODE:-train}; REMAT=${REMAT:-nothing_saveable}
BATCH_FALLBACK=${BATCH_FALLBACK:-}
case $MODE in
  smoke) CFG=pi05_robomme_0920_v1_smoke; EXP=${EXP:-smoke_0920_v1}; WANDB=0 ;;
  probe) CFG=pi05_robomme_0920_v1_probe; EXP=${EXP:-probe_0920_v1_b$BATCH}; WANDB=0 ;;
  *)     CFG=pi05_robomme_0920_v1; EXP=${EXP:-robomme_0920_v1}; WANDB=${WANDB:-1} ;;
esac
status=$ROOT/robomme/logs/train_0920_v1_status.log
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
gpu_busy() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; }
wait_gpu() { for i in $(seq 1 240); do [ "$(gpu_busy)" = "0" ] && return 0; sleep 30; done; return 1; }
log "start mode=$MODE host=$(hostname) job=$JOB batch=$BATCH fallback='$BATCH_FALLBACK' workers=$WORKERS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION remat=$REMAT cfg=$CFG exp=$EXP code=$(git -C $ROOT rev-parse --short HEAD)"
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
run_once() { OPENPI_0920_V1_BATCH=$1 OPENPI_0920_REMAT=$REMAT WANDB=$WANDB JOB=$JOB GRES=4 VISIBLE=0,1,2,3 GPUS=4 BATCH=$1 ACCUM=1 CPUS=32 bash cluster_robomme/run_train.sh "$CFG" "$EXP" --num-workers $WORKERS; }
for b in $BATCH $BATCH_FALLBACK; do
  log "attempt batch=$b remat=$REMAT exp=$EXP"
  run_once $b
  st=$(tail -1 "$ROOT/robomme/logs/train_${EXP}_status.log")
  if echo "$st" | grep -q "^exit=0"; then log "end mode=$MODE batch=$b: $st"; exit 0; fi
  if tail -400 "$ROOT/robomme/logs/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
    log "batch $b ran out of memory ($st)"; rm -rf "$ROOT/robomme/checkpoints/$CFG/$EXP"; sleep 30; wait_gpu || { log "cards busy after OOM"; exit 1; }; continue
  fi
  log "end mode=$MODE batch=$b (not an OOM): $st"; exit 1
done
log "end mode=$MODE: every batch ran out of memory"; exit 1
