#!/usr/bin/env bash
# Full smoke chain on the user's test job 17405067 (1xH100, iris-hgx-1), run ON iris-hgx-1:
#   nohup setsid bash cluster_robomme/smoke_all_testjob.sh > robomme/logs/smoke_all.out 2>&1 &
# 1. CPU loader smoke (both configs)  2. step-0 GPU smoke of the memory config (plain7000 warm start, batch 1)
# 3. step-0 GPU smoke of the KI base config (batch 2). Shares the card with the 1 GB keep-alive + other processes.
set -u
JOB="${JOB:-17405067}"; GPU="${GPU:-0}"
root=/iris/u/kewalk/memory_project_robomme; cd "$root/openpi" || exit 2
source cluster_robomme/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
status="$root/robomme/logs/smoke_all.log"
echo "smoke chain start $(date) host=$(hostname)" | tee -a "$status"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 \
  .venv/bin/python cluster_robomme/smoke_loader.py > "$root/robomme/logs/smoke_loader_node.log" 2>&1
rc=$?; echo "loader smoke exit=$rc $(date)" | tee -a "$status"
[ $rc -eq 0 ] || exit $rc
JOB=$JOB GPU=$GPU bash cluster_robomme/smoke_step0.sh pi05_robomme_mem_PickXtimes_B_plain7000 1
echo "mem smoke done $(date)" | tee -a "$status"
JOB=$JOB GPU=$GPU bash cluster_robomme/smoke_step0.sh pi05_robomme_PickXtimes_base_ki 2
echo "base smoke done $(date)" | tee -a "$status"
echo "smoke chain end $(date)" | tee -a "$status"
