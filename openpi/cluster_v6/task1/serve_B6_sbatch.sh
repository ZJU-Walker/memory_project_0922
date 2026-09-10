#!/bin/bash
# Policy server for a v6 task1 checkpoint on ONE iris-partition GPU (user 2026-09-10 13:39: "get another iris partition
# l40s and let's serve the 1600").   ssh sc sbatch --gres=gpu:l40s:1 cluster_v6/task1/serve_B6_sbatch.sh
# Env: CK (checkpoint dir with params/), CFG, PORT (8000), NUM_STEPS (10 flow steps). Prints the node IP for the client.
#SBATCH --job-name=v6_task1_serve
#SBATCH --output=/iris/u/kewalk/memory_project_v6/v6/logs/serve_sbatch-%j.out
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=12
#SBATCH --account=iris
set -u
CK="${CK:-/iris/u/kewalk/memory_project_v6/v6/checkpoints/pi05_yam_mem_v6_task1B6/v6_task1B6_20260909_r3/1600}"
CFG="${CFG:-pi05_yam_mem_v6_task1B6}"; PORT="${PORT:-8000}"; NUM_STEPS="${NUM_STEPS:-10}"
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1
cd /iris/u/kewalk/memory_project_v6/openpi || exit 2
source cluster_v6/env.sh >/dev/null 2>&1
ip=$(hostname -I | awk '{print $1}')
echo "==== serve job $SLURM_JOB_ID on $SLURMD_NODENAME ($ip) $(date) ck=$CK cfg=$CFG port=$PORT num_steps=$NUM_STEPS ===="
nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader
if ! .venv/bin/python -c "import jax; d=jax.devices(); assert d[0].platform=='gpu', d; print('jax ok', d)"; then
  echo "JAX cannot use the allocated GPU on $SLURMD_NODENAME -> exiting (resubmit with --exclude=$SLURMD_NODENAME)"; exit 7
fi
log=/iris/u/kewalk/memory_project_v6/v6/diagnostics/server_v6_$(basename $(dirname $(dirname $CK)))_$(basename $CK)_job${SLURM_JOB_ID}.log
echo "SERVER ADDRESS: $ip:$PORT  (log $log)"
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 PYTHONPATH=scripts \
  .venv/bin/python -u scripts/serve_yam_memory.py --dir "$CK" --config "$CFG" --port "$PORT" --warmup --num-steps "$NUM_STEPS" 2>&1 | tee -a "$log"
echo "server exited $(date)"
