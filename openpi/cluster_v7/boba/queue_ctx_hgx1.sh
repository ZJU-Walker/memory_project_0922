#!/usr/bin/env bash
# Phase-context ablation (cluster_v7/README.md §2) on the 2xH100 of job 17356154 (iris-hgx-1), sequential:
# ctx_none -> ctx_prev -> ctx_state -> ctx_both, FSDP over both GPUs, global batch 16, 5k steps each from the
# boba base 9999. The two long-prompt variants use gradient accumulation 2 (the base already filled the H100s
# at 208 tokens). Run ON iris-hgx-1:
#   nohup setsid bash cluster_v7/boba/queue_ctx_hgx1.sh > v7/logs/queue_ctx_hgx1.out 2>&1 &
set -u
JOB=17356154; TAG=20260912
root=/iris/u/kewalk/memory_project_v7
cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/kill_placeholders_on_node.sh $JOB
sleep 10
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --gres=gpu:2 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
run() { JOB=$JOB GRES=2 VISIBLE=0,1 GPUS=2 BATCH=16 ACCUM=$2 CPUS=16 bash cluster_v7/run_train.sh "$1" "${1#pi05_yam_boba0911_}_${TAG}_r1"; }
run pi05_yam_boba0911_ctx_none 1
run pi05_yam_boba0911_ctx_prev 1
run pi05_yam_boba0911_ctx_state 2
run pi05_yam_boba0911_ctx_both 2
echo "queue done $(date)"
