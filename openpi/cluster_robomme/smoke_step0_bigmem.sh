#!/usr/bin/env bash
# Step-0 GPU smoke of a RoboMME config on ONE GPU of a Slurm job (default: the user's test job 17405067, 1xH100 on
# iris-hgx-1): loads the warm-start checkpoint, compiles, runs 3 optimizer updates at a tiny batch, writes the
# initialization audit + a throwaway checkpoint under robomme/checkpoints/<config>/smoke_<tag>/. Run ON the job's node:
#   JOB=17405067 GPU=0 bash cluster_robomme/smoke_step0.sh pi05_robomme_mem_PickXtimes_B [batch]
# Shares the card (XLA_PYTHON_CLIENT_PREALLOCATE=false, MEM_FRACTION 0.6); the job's 1 GB keep-alive stays.
set -u
config="$1"; batch="${2:-2}"; JOB="${JOB:-17405067}"; GPU="${GPU:-0}"; GRES="${GRES:-1}"
root=/iris/u/kewalk/memory_project_robomme; cd "$root/openpi" || exit 2
source cluster_robomme/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
tag="$(date +%m%d_%H%M)"; exp="smoke_${tag}"
log="$root/robomme/logs/smoke_step0_${config}_${tag}_bigmem.log"
echo "smoke start $(date) host=$(hostname) job=$JOB gpu=$GPU config=$config batch=$batch exp=$exp" | tee "$log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$GRES" \
  env CUDA_VISIBLE_DEVICES="$GPU" \
  cluster_robomme/train.sh "$config" --exp-name "$exp" --batch-size "$batch" --gradient-accumulation-steps 1 --fsdp-devices 1 \
  --num-train-steps 3 --save-interval 1000 --log-interval 1 --num-workers 2 --no-wandb-enabled --overwrite >> "$log" 2>&1
echo "smoke exit=$? $(date)" | tee -a "$log"
