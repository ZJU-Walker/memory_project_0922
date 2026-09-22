#!/usr/bin/env bash
# Generate 100 extra BinFill expert demos with the benchmark recorder on the 2 x H100 job (09-21, user: "before you train,
# you can use the Robomme provided tool to generate another 100 training demos"). Run ON iris-hgx-1:
#   setsid nohup bash robomme/logs/gen_binfill_demos.sh > robomme/logs/gen_binfill.log 2>&1 &
# One --overlap step with BOTH GPUs of the job (a smaller --gres lands on physical GPU 0); the driver pins 4 workers per GPU.
set -uo pipefail
export HOME=/iris/u/kewalk
JOB=${JOB:-17489557}
ROOT=/iris/u/kewalk/memory_project_0920
BENCH=${ROBOMME_BENCHMARK_ROOT:-/iris/u/kewalk/robomme_benchmark}
OUT=${OUT:-/scr/kewalk/robomme_0920/gen/BinFill}
WORKERS=${WORKERS:-8}
echo "launch $(date '+%m/%d %H:%M') host=$(hostname) job=$JOB out=$OUT workers=$WORKERS"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=${CPUS:-32} --gres=gpu:2 \
  env HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      ROBOMME_BENCHMARK_ROOT="$BENCH" PYTHONPATH="$BENCH/src" \
  bash -c "
    set -u
    echo step host=\$(hostname) visible=\${CUDA_VISIBLE_DEVICES:-unset}; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
    $BENCH/.venv/bin/python -c 'import sys; sys.path[:0] = [\"$BENCH\", \"$BENCH/src\"]; from tests._shared import dataset_generation as g; import robomme; print(\"import ok:\", g.__file__, g.MAX_SEED_ATTEMPTS)' || exit 3
    exec $BENCH/.venv/bin/python -u $ROOT/openpi/cluster_robomme/generate_robomme_demos.py --env BinFill --out $OUT --episodes 100 --seed-base 24000 --workers $WORKERS --gpus 0,1
  "
echo "exit=$? $(date '+%m/%d %H:%M')"
