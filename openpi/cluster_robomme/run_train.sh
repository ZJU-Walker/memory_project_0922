#!/usr/bin/env bash
# RoboMME training as an --overlap step of Slurm job $JOB with explicit GPU pinning inside the job:
#   JOB=<job> GRES=<gpus of the job> VISIBLE=<idx,idx> GPUS=<fsdp devices> [BATCH=16] [ACCUM=1] [CPUS=8] \
#     cluster_robomme/run_train.sh <config-name> <exp-name> [extra train.py args]
# Run ON the node of the job. `--gres` must request ALL GPUs of the job and the run is pinned with
# CUDA_VISIBLE_DEVICES (an --overlap step with a smaller --gres always lands on physical GPU 0). Resume policy:
# numeric checkpoint in the experiment dir -> --resume; dir without one -> --overwrite.
# Logs: robomme/logs/train_<exp>.log and train_<exp>_status.log (appended).
set -u
config="$1"; exp="$2"; shift 2
JOB="${JOB:?job id}"; gres="${GRES:?number of GPUs of the job}"; visible="${VISIBLE:?CUDA_VISIBLE_DEVICES list}"
gpus="${GPUS:-$(echo "$visible" | tr ',' '\n' | wc -l)}"
batch="${BATCH:-16}"; accum="${ACCUM:-1}"; cpus="${CPUS:-8}"; wandb="${WANDB:-0}"  # WANDB=1 turns W&B on (user 09-15 15:29); project from the config, ~/.netrc on the node
root="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"  # portable: the tree this script lives in
cd "$root/openpi" || exit 2
source cluster_robomme/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
logs="${LOGS:-$root/robomme/logs}"; mkdir -p "$logs"  # LOGS / CKBASE: another line's log and checkpoint roots (beans: beans/logs, beans/checkpoints)
ckdir="${CKBASE:-$root/robomme/checkpoints}/$config/$exp"
mode=fresh; extra=()
if [ -d "$ckdir" ]; then
  if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi
fi
echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gpus=$visible fsdp=$gpus config=$config exp=$exp batch=$batch accum=$accum wandb=$wandb mode=$mode code=$(git -C $root rev-parse --short HEAD) extra=$*" >> "$logs/train_${exp}_status.log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --gres=gpu:"$gres" \
  env CUDA_VISIBLE_DEVICES="$visible" \
  cluster_robomme/train.sh "$config" --exp-name "$exp" --batch-size "$batch" --gradient-accumulation-steps "$accum" --fsdp-devices "$gpus" \
  $( [ "${WANDB:-0}" = "1" ] && echo --wandb-enabled || echo --no-wandb-enabled ) "${extra[@]}" "$@" >> "$logs/train_${exp}.log" 2>&1
echo "exit=$? $(date +%m/%d\ %H:%M)" >> "$logs/train_${exp}_status.log"
