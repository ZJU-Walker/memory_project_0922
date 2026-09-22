#!/usr/bin/env bash
# 0920 BinFill on the 2 x H100 (job $JOB, default 17489557; user 09-21 02:30 "just use 2h100 as you want"): the 0920_v0
# recipe on BinFill200 (released 100 + 100 generated demos), config pi05_robomme_0920_binfill_v0, W&B project robomme_0920.
# Run ON iris-hgx-1 via train_binfill_ctl.sh (setsid/nohup). The 1 GB train_hs keep-alive of the job is never touched.
# MODE=smoke runs the _smoke config (3 updates, no W&B). REMAT=dots_saveable first, automatic fallback to nothing_saveable
# on an out-of-memory exit (same as the H200 runner v2). nvidia-smi is read INSIDE the job step (a plain shell on the node
# lands in the user's newest job and sees only that job's card).
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
JOB=${JOB:-17489557}; BATCH=${BATCH:-8}; WORKERS=${WORKERS:-16}; MODE=${MODE:-train}
if [ "$MODE" = smoke ]; then CFG=pi05_robomme_0920_binfill_v0_smoke; EXP=${EXP:-smoke_0920_binfill_v0}; WANDB=0; else CFG=pi05_robomme_0920_binfill_v0; EXP=${EXP:-robomme_0920_binfill_v0}; WANDB=${WANDB:-1}; fi
status=$ROOT/robomme/logs/train_binfill_status.log
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
gpu_busy() { srun --jobid=$JOB --overlap --gres=gpu:2 nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; }
wait_gpu() { for i in $(seq 1 240); do [ "$(gpu_busy)" = "0" ] && return 0; sleep 30; done; return 1; }
log "start mode=$MODE host=$(hostname) job=$JOB batch=$BATCH workers=$WORKERS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION cfg=$CFG exp=$EXP dataset=${OPENPI_0920_BINFILL_DATASET:-BinFill200} code=$(git -C $ROOT rev-parse --short HEAD)"
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
REMAT=${REMAT:-dots_saveable}
run_once() { OPENPI_0920_REMAT=$1 WANDB=$WANDB JOB=$JOB GRES=2 VISIBLE=0,1 GPUS=2 BATCH=$BATCH ACCUM=${ACCUM:-1} CPUS=${CPUS:-24} bash cluster_robomme/run_train.sh "$CFG" "$EXP" --num-workers $WORKERS; }
log "attempt remat=$REMAT"
run_once "$REMAT"
last_exit=$(tail -1 "$ROOT/robomme/logs/train_${EXP}_status.log" | grep -o "^exit=[0-9]*" | cut -d= -f2)
if [ "$REMAT" != nothing_saveable ] && [ "${last_exit:-0}" != 0 ] && tail -400 "$ROOT/robomme/logs/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
  log "remat=$REMAT ran out of memory; falling back to nothing_saveable"
  sleep 30; wait_gpu || { log "cards busy after the OOM attempt"; exit 1; }
  run_once nothing_saveable
fi
log "end mode=$MODE: $(tail -1 $ROOT/robomme/logs/train_${EXP}_status.log)"
