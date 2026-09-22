#!/usr/bin/env bash
# Memory analysis of the v1 train step at several batch sizes (+ v0 at 32 as the calibration point: it ran) on the 4 x H200
# node, in parallel processes, no preallocation. ON iris-hgx-2 inside job $JOB. Results -> robomme/logs/aot_memory_<cfg>_b<B>.log
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
source cluster_robomme/env.sh >/dev/null 2>&1
JOB=${JOB:-17422727}
run() { srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:4 env CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .venv/bin/python $ROOT/robomme/logs/aot_memory.py $1 $2 $ROOT/robomme/logs/batch_spec_$1.pkl 4 > $ROOT/robomme/logs/aot_memory_$1_b$2.log 2>&1; }
echo "start $(date +%H:%M:%S)"
run pi05_robomme_0920_v0_probe 32 &
for b in ${BATCHES:-32 40 48 56}; do run pi05_robomme_0920_v1_probe $b & done
wait
grep -h "^RESULT\|Error\|error:" $ROOT/robomme/logs/aot_memory_*_b*.log | cut -c1-300
echo "done $(date +%H:%M:%S)"
