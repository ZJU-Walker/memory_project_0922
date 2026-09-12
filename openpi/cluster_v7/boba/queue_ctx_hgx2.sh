#!/usr/bin/env bash
# Phase-context ablation (cluster_v7/README.md §2) on the 4xH200 of job 17356093 (iris-hgx-2): two queues in
# parallel, each FSDP over 2 GPUs, global batch 16, 5k steps continuing the boba base 9999.
#   queue A (GPUs 0,1): ctx_none -> ctx_prev      queue B (GPUs 2,3): ctx_state -> ctx_both
# Run ON iris-hgx-2:  nohup setsid bash cluster_v7/boba/queue_ctx_hgx2.sh > v7/logs/queue_ctx_hgx2.out 2>&1 &
set -u
JOB=17356093; TAG=20260912
root=/iris/u/kewalk/memory_project_v7
cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/kill_placeholders_on_node.sh $JOB
sleep 10
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --gres=gpu:4 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
queue() {  # queue <visible> <config>...
  local visible="$1"; shift
  for cfg in "$@"; do
    JOB=$JOB GRES=4 VISIBLE=$visible GPUS=2 BATCH=16 ACCUM=1 CPUS=8 bash cluster_v7/run_train.sh "$cfg" "${cfg#pi05_yam_boba0911_}_${TAG}_r1"
  done
}
queue 0,1 pi05_yam_boba0911_ctx_none pi05_yam_boba0911_ctx_prev &
queue 2,3 pi05_yam_boba0911_ctx_state pi05_yam_boba0911_ctx_both &
wait
echo "queues done $(date)"
