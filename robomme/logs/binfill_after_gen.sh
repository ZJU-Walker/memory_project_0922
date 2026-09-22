#!/usr/bin/env bash
# After the 100 extra BinFill demos are generated (run ON iris-hgx-1): (1) convert released + generated into BinFill200,
# (2) probe the batch size on the free H100 pair with the released-only BinFill100 spec (31 updates each, one 42-tick
# shape, dots_saveable): batch 8 -> 12 -> 16, stopping at the first out-of-memory. Status in robomme/logs/binfill_after_gen.log.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
JOB=${JOB:-17489557}; GEN=/scr/kewalk/robomme_0920/gen/BinFill
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*"; }
n=0; until [ -f "$GEN/summary.json" ] || [ $n -ge 360 ]; do sleep 10; n=$((n+1)); done
[ -f "$GEN/summary.json" ] || { log "generation summary missing after $((n*10))s"; exit 1; }
log "generation: $(cat $GEN/summary.json | tr -d '\n' | cut -c1-300)"
ok=$(.venv/bin/python -c "import json; d=json.load(open('$GEN/summary.json')); print(int(d['succeeded']==d['requested']==100))")
[ "$ok" = 1 ] || { log "generation incomplete; not converting"; exit 1; }
source cluster_robomme/env.sh >/dev/null 2>&1
log "BinFill200 conversion starting (background, CPU)"
setsid nohup srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 .venv/bin/python -u cluster_robomme/prepare_robomme_h5_to_lerobot.py \
  --env BinFill --name BinFill200 --released /iris/u/kewalk/robomme_benchmark/data/robomme_data_h5/record_dataset_BinFill.h5 \
  --extra-dir $GEN/hdf5_files --expect-episodes 200 --project-root $ROOT --scratch-root /scr/kewalk/robomme_0920 \
  > $ROOT/robomme/logs/prepare_binfill200.log 2>&1 < /dev/null &
spec=$ROOT/robomme/metadata/BinFill100/prepared_official.json
n=0; until [ -f "$spec" ] || [ $n -ge 120 ]; do sleep 15; n=$((n+1)); done
[ -f "$spec" ] || { log "BinFill100 spec missing; no batch probe"; exit 1; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
for b in 8 12 16; do
  exp=probe_binfill_b$b
  log "probe batch $b start"
  OPENPI_0920_BINFILL_DATASET=BinFill100 OPENPI_0920_REMAT=dots_saveable WANDB=0 JOB=$JOB GRES=2 VISIBLE=0,1 GPUS=2 BATCH=$b ACCUM=1 CPUS=16 \
    bash cluster_robomme/run_train.sh pi05_robomme_0920_binfill_v0_probe $exp --num-workers 8
  st=$(tail -1 $ROOT/robomme/logs/train_${exp}_status.log)
  t10=$(grep -E "Progress on: 10\.00it" $ROOT/robomme/logs/train_${exp}.log | tail -1 | cut -c1-12)
  t30=$(grep -E "Progress on: 30\.00it" $ROOT/robomme/logs/train_${exp}.log | tail -1 | cut -c1-12)
  log "probe batch $b: $st | t(10)=$t10 t(30)=$t30"
  if grep -q "RESOURCE_EXHAUSTED" $ROOT/robomme/logs/train_${exp}.log; then log "probe batch $b OOM; stopping the chain"; break; fi
done
log "after-gen chain done"
