#!/bin/bash
# Launch the boba pi05+KI base run on the 2xH100 of job 17356154 (iris-hgx-1; user 2026-09-12 02:00 "use the free
# h100 to train the base pi05 policy, and keep every 5000 train till 10000"). Run ON iris-hgx-1:
#   nohup setsid bash cluster_v6/boba/launch_base_train_hgx1.sh <exp-name> > v6/logs/launch_boba_<exp>.out 2>&1 &
# 1. kills OUR placeholder of that job (marker GPU_PLACEHOLDER=gpu_placeholder_marker_17356154; never the 1 GB
#    train_hs.py keep-alives), 2. starts cluster_v6/run_train_hgx1.sh as an --overlap step with FSDP over both GPUs
#    (global batch 16). Logs: v6/logs/train_<exp>.log / train_<exp>_status.log.
set -u
exp="${1:?exp name}"
JOB=17356154
cd /iris/u/kewalk/memory_project_v6/openpi || exit 2
export HOME=/iris/u/kewalk
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/kill_placeholders_on_node.sh $JOB
sleep 10
srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --gres=gpu:2 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
JOB=$JOB GPUS=2 BATCH=16 ACCUM=1 CPUS=16 bash cluster_v6/run_train_hgx1.sh pi05_yam_boba0911_base "$exp"
