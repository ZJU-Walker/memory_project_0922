#!/usr/bin/env bash
# v6 training on the 4xH100 of Slurm job $JOB (default 17315830, iris-hgx-1; user 2026-09-08 19:47: "for test and
# training use 17315830 4h100 ... keep the 1gb alive" = never touch the train_hs.py keep-alive of that job).
#   [JOB=17315830] [GPUS=4] [BATCH=8] [ACCUM=1] cluster_v6/run_train_hgx1.sh <config-name> <exp-name> [extra train.py args]
# Run ON the node (ssh iris-hgx-1 lands in the job's cgroup): the payload is an --overlap step of the job.
# Same recipe as the v5 beans A9/B9 runs (cluster_v5/run_train_h200.sh with GPUS=4 BATCH=8: FSDP over the 4 GPUs,
# global batch 8). Resume policy: numeric checkpoint in the experiment dir -> --resume; dir without one -> --overwrite.
# Writes v6/logs/train_<exp>_status.log and v6/logs/train_<exp>.log (appended).
set -u
config="$1"; exp="$2"; shift 2
batch="${BATCH:-8}"; accum="${ACCUM:-1}"; JOB="${JOB:-17315830}"
gpus="${GPUS:-4}"; visible=$(seq -s, 0 $((gpus - 1)))
root=/iris/u/kewalk/memory_project_v6
cd "$root/openpi" || exit 2
source cluster_v6/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
logs="$root/v6/logs"; mkdir -p "$logs"
ckdir="$root/v6/checkpoints/$config/$exp"
mode=fresh; extra=()
if [ -d "$ckdir" ]; then
  if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi
fi
job=$(grep -oE 'job_[0-9]+' /proc/self/cgroup | sort -u | tr '\n' ' ')
echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=$job step-of=$JOB gpus=$gpus config=$config exp=$exp batch=$batch accum=$accum mode=$mode code=$(git -C $root rev-parse --short HEAD) extra=$*" >> "$logs/train_${exp}_status.log"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:"$gpus" \
  env CUDA_VISIBLE_DEVICES="$visible" \
  cluster_v6/train.sh "$config" --exp-name "$exp" --batch-size "$batch" --gradient-accumulation-steps "$accum" --fsdp-devices "$gpus" \
  --no-wandb-enabled "${extra[@]}" "$@" >> "$logs/train_${exp}.log" 2>&1
echo "exit=$? $(date +%m/%d\ %H:%M)" >> "$logs/train_${exp}_status.log"
