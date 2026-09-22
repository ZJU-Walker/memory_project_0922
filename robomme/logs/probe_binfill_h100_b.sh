#!/usr/bin/env bash
# Second BinFill batch probe on the 2 x H100 pair (job 17489557) after batch 8 + dots_saveable OOMed at 42 ticks (03:42,
# 60.8 GB allocation): batch 8 with the default remat (nothing_saveable), then batch 6, then batch 4 + dots_saveable
# (the known-fitting 40-tick setting); stops at the first probe that fits. 31 updates each, one 42-tick shape, BinFill100.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
JOB=${JOB:-17489557}
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*"; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
for spec in "8 nothing_saveable" "6 nothing_saveable" "4 dots_saveable"; do
  set -- $spec; b=$1; remat=$2; exp=probe_binfill_b${b}_${remat%%_*}
  log "probe batch $b remat $remat start"
  OPENPI_0920_BINFILL_DATASET=BinFill100 OPENPI_0920_REMAT=$remat WANDB=0 JOB=$JOB GRES=2 VISIBLE=0,1 GPUS=2 BATCH=$b ACCUM=1 CPUS=16 \
    bash cluster_robomme/run_train.sh pi05_robomme_0920_binfill_v0_probe $exp --num-workers 8
  st=$(tail -1 $ROOT/robomme/logs/train_${exp}_status.log)
  t10=$(grep -E "Progress on: 10\.00it" $ROOT/robomme/logs/train_${exp}.log | tail -1 | cut -c1-12)
  t30=$(grep -E "Progress on: 30\.00it" $ROOT/robomme/logs/train_${exp}.log | tail -1 | cut -c1-12)
  log "probe batch $b remat $remat: $st | t(10)=$t10 t(30)=$t30"
  if grep -q "RESOURCE_EXHAUSTED" $ROOT/robomme/logs/train_${exp}.log; then log "batch $b remat $remat OOM"; continue; fi
  if echo "$st" | grep -q "^exit=0"; then log "FITS: batch $b remat $remat"; echo "$b $remat" > $ROOT/robomme/logs/probe_binfill_fit.txt; break; fi
  log "batch $b remat $remat failed for another reason; see train_${exp}.log"; break
done
log "probe chain b done"
