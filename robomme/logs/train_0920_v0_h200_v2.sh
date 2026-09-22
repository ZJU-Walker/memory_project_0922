#!/usr/bin/env bash
# 0920_v0 on the 4 x H200 (job $JOB, default 17422727 offered by the user 09-20 22:11 "for gpu test"): ONE self-write run
# from the original pi05 base, W&B project robomme_0920. Run ON the node via train_0920_ctl.sh (setsid/nohup). The 1 GB
# train_hs keep-alives are never touched. MODE=smoke runs pi05_robomme_0920_v0_smoke (3 updates, no W&B) instead.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
JOB=${JOB:-17422727}; BATCH=${BATCH:-8}; WORKERS=${WORKERS:-16}; MODE=${MODE:-train}
if [ "$MODE" = smoke ]; then CFG=pi05_robomme_0920_v0_smoke; EXP=${EXP:-smoke_0920_v0}; WANDB=0; else CFG=pi05_robomme_0920_v0; EXP=${EXP:-robomme_0920_v0}; WANDB=${WANDB:-1}; fi  # WANDB=0 for timing probes
status=$ROOT/robomme/logs/train_0920_status.log
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
gpu_busy() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; }
wait_gpu() { for i in $(seq 1 240); do [ "$(gpu_busy)" = "0" ] && return 0; sleep 30; done; return 1; }
log "start mode=$MODE host=$(hostname) job=$JOB batch=$BATCH workers=$WORKERS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION cfg=$CFG exp=$EXP code=$(git -C $ROOT rev-parse --short HEAD)"
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
# 09-21 01:45: remat policy for the resume. dots_saveable measured 11.22 vs 12.27 s/update on the 2 x H100 (batch 4; the
# everything_saveable policy OOMed there). Its extra activation memory at 8 windows per H200 is untested, so the first
# attempt uses it and an OOM exit falls back to the default policy automatically (a failed attempt costs ~10 min).
REMAT=${REMAT:-dots_saveable}
run_once() { OPENPI_0920_REMAT=$1 WANDB=$WANDB JOB=$JOB GRES=4 VISIBLE=0,1,2,3 GPUS=4 BATCH=$BATCH ACCUM=1 CPUS=32 bash cluster_robomme/run_train.sh "$CFG" "$EXP" --num-workers $WORKERS; }
log "attempt remat=$REMAT"
run_once "$REMAT"
last_exit=$(tail -1 "$ROOT/robomme/logs/train_${EXP}_status.log" | grep -o "^exit=[0-9]*" | cut -d= -f2)
if [ "$REMAT" != nothing_saveable ] && [ "${last_exit:-0}" != 0 ] && tail -400 "$ROOT/robomme/logs/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
  log "remat=$REMAT ran out of memory; falling back to nothing_saveable"
  sleep 30; wait_gpu || { log "cards busy after the OOM attempt"; exit 1; }
  run_once nothing_saveable
fi
log "end mode=$MODE: $(tail -1 $ROOT/robomme/logs/train_${EXP}_status.log)"
