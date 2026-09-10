#!/usr/bin/env bash
# Fresh v6.5 line (user 2026-09-09 23:26: "start fresh ... first A and then B, A 200 steps, B save every 200"):
# run stage A6 (oracle writes, from the beans B9 ckpt) to its step-200 checkpoint, then stage B6 (own writes) from it.
# Run ON iris-hgx-1: JOB=17356154 GPUS=2 BATCH=8 ACCUM=1 setsid nohup bash cluster_v6/task1/chain_A6_B6_hgx1.sh &
set -u
export JOB="${JOB:-17356154}" GPUS="${GPUS:-2}" BATCH="${BATCH:-8}" ACCUM="${ACCUM:-1}"
root=/iris/u/kewalk/memory_project_v6; cd "$root/openpi" || exit 2
A_EXP="${A_EXP:-v6_task1A6_20260909_r1}"; B_EXP="${B_EXP:-v6_task1B6_20260909_r3}"
status="$root/v6/logs/chain_A6_B6_$(date +%Y%m%d_%H%M).log"
echo "chain start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gpus=$GPUS batch=$BATCH accum=$ACCUM A=$A_EXP B=$B_EXP" | tee -a "$status"
bash cluster_v6/run_train_hgx1.sh pi05_yam_mem_v6_task1A6 "$A_EXP"
a_ck="$root/v6/checkpoints/pi05_yam_mem_v6_task1A6/$A_EXP/200/params"
if [ ! -d "$a_ck" ]; then echo "A6 step-200 checkpoint missing ($a_ck) $(date +%m/%d\ %H:%M); B not launched" | tee -a "$status"; exit 3; fi
echo "A6 done $(date +%m/%d\ %H:%M): $a_ck; launching B6 $B_EXP" | tee -a "$status"
export OPENPI_V6_TASK1_A6_PARAMS="v6/checkpoints/pi05_yam_mem_v6_task1A6/$A_EXP/200/params"  # project-relative (project_path rejects absolute)
bash cluster_v6/run_train_hgx1.sh pi05_yam_mem_v6_task1B6 "$B_EXP"
echo "chain end $(date +%m/%d\ %H:%M)" | tee -a "$status"
