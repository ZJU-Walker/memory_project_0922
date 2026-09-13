#!/usr/bin/env bash
# v7 boba memory, two-phase labels, ONE stage (own write timing + label content straight from the boba base) on GPUs 0,1
# of the 4xH200 job 17403682 (user 2026-09-13 15:11: "use only 2 h200 so gpu0 and 1 for testing and training").
# --gres must request the job's full 4 GPUs (an --overlap step with a smaller gres lands on physical GPU 0); the run is
# pinned with CUDA_VISIBLE_DEVICES=0,1, FSDP 2, batch 4 (two windows per GPU). Run ON iris-hgx-2:
#   nohup setsid bash cluster_v7/boba/run_mem_2gpu_v2.sh > v7/logs/run_mem_2gpu_v2_r1.out 2>&1 &
set -u
export JOB="${JOB:-17403682}" GRES="${GRES:-4}" VISIBLE="${VISIBLE:-0,1}" GPUS="${GPUS:-2}" BATCH="${BATCH:-4}" ACCUM="${ACCUM:-1}" CPUS="${CPUS:-16}"
root=/iris/u/kewalk/memory_project_v7; cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
CONFIG="${CONFIG:-pi05_yam_mem_v7_boba2B}"; EXP="${EXP:-v7_boba2B_20260913_r1}"
status="$root/v7/logs/run_mem_2gpu_v2.log"
echo "start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gres=$GRES gpus=$VISIBLE fsdp=$GPUS batch=$BATCH config=$CONFIG exp=$EXP code=$(git -C $root rev-parse --short HEAD)" | tee -a "$status"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"$GRES" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$status"
bash cluster_v7/run_train.sh "$CONFIG" "$EXP"
echo "end $(date +%m/%d\ %H:%M): $(tail -1 $root/v7/logs/train_${EXP}_status.log)" | tee -a "$status"
