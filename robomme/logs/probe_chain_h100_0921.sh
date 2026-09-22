#!/usr/bin/env bash
# Sequential timing probes on the 2 x H100 pair (job 17489557), 31 updates each, single 40-tick shape, hoist ON in all:
#   remat_dots  = remat_policy dots_saveable | remat_all = everything_saveable | xla_lhs = XLA latency-hiding scheduler flags
set -u
cd /iris/u/kewalk/memory_project_0920/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
run() { local exp=$1; shift; env "$@" WANDB=0 JOB=17489557 GRES=2 VISIBLE=0,1 GPUS=2 BATCH=4 ACCUM=1 CPUS=16 bash cluster_robomme/run_train.sh pi05_robomme_0920_v0_probe "$exp" --num-workers 8; }
run probe_h100_remat_dots OPENPI_0920_REMAT=dots_saveable
run probe_h100_remat_all OPENPI_0920_REMAT=everything_saveable
run probe_h100_xla_lhs XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_while_loop_double_buffering=true"
echo "chain done $(date +%m/%d\ %H:%M)" >> /iris/u/kewalk/memory_project_0920/robomme/logs/probe_chain_h100_0921.status
