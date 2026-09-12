#!/usr/bin/env bash
# v7 boba memory line on the 4xH200 of job 17403682 (iris-hgx-2; cluster_v7/README.md §3).
# History: r1 = 2xH200 job 17403858, batch 4, 28 s/update, killed with the job at update 235 (user 15:56). r2 = this job,
# batch 4 = one window per GPU, only 22 s/update (the 60-step recurrence is sequential), stopped at update ~20.
# r3 (user 16:25, "batch 8, half the updates"): global batch 8 = two windows per GPU (~28 s/update, twice the samples),
# stage A 250 updates (oracle writes, from the boba base 9999) then stage B 1500 updates (own writes, from A/250).
# The job's 1 GB train_hs.py keep-alives stay. Run ON iris-hgx-2:
#   nohup setsid bash cluster_v7/boba/chain_mem_hgx2_4gpu.sh > v7/logs/chain_mem_hgx2_4gpu_r3.out 2>&1 &
set -u
export JOB="${JOB:-17403682}" GRES="${GRES:-4}" VISIBLE="${VISIBLE:-0,1,2,3}" GPUS="${GPUS:-4}" BATCH="${BATCH:-8}" ACCUM="${ACCUM:-1}" CPUS="${CPUS:-24}"
root=/iris/u/kewalk/memory_project_v7; cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
A_EXP="${A_EXP:-v7_bobaA_20260912_r3}"; B_EXP="${B_EXP:-v7_bobaB_20260912_r3}"; A_STEP="${A_STEP:-250}"
status="$root/v7/logs/chain_mem_hgx2_4gpu.log"
echo "chain start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gres=$GRES gpus=$VISIBLE fsdp=$GPUS batch=$BATCH accum=$ACCUM A=$A_EXP B=$B_EXP code=$(git -C $root rev-parse --short HEAD)" | tee -a "$status"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"$GRES" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$status"
bash cluster_v7/run_train.sh pi05_yam_mem_v7_bobaA "$A_EXP"
a_ck="$root/v7/checkpoints/pi05_yam_mem_v7_bobaA/$A_EXP/$A_STEP/params"
if [ ! -d "$a_ck" ]; then echo "A step-$A_STEP checkpoint missing ($a_ck) $(date +%m/%d\ %H:%M); B not launched" | tee -a "$status"; exit 3; fi
echo "A done $(date +%m/%d\ %H:%M): $a_ck; launching B $B_EXP" | tee -a "$status"
export OPENPI_V7_BOBA_A_PARAMS="v7/checkpoints/pi05_yam_mem_v7_bobaA/$A_EXP/$A_STEP/params"  # project-relative (project_path rejects absolute)
bash cluster_v7/run_train.sh pi05_yam_mem_v7_bobaB "$B_EXP"
echo "chain end $(date +%m/%d\ %H:%M)" | tee -a "$status"
