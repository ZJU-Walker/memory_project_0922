#!/usr/bin/env bash
# Stop a running v6 task1 stage-A queue/runner/train (never anything else of job 17315830) and re-arm the queue.
# A script file on purpose: an inline `bash -c` with these pgrep patterns matches its own command line and kills itself.
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs
pat="scripts/train.py pi05_yam_mem_v[6]|queue_task1A_hgx[1]|run_train_hgx1.s[h]|cluster_v6/train.s[h]"
echo "--- leftovers:"; pgrep -u kewalk -af "$pat" | cut -c1-120
left=$(pgrep -u kewalk -f "$pat")
if [ -n "$left" ]; then
  echo "stopping: $left"; kill -TERM $left 2>/dev/null; sleep 12
  l2=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v[6]"); [ -n "$l2" ] && { echo "SIGKILL $l2"; kill -KILL $l2; sleep 5; }
fi
echo "keep-alive 3743806 alive: $([ -d /proc/3743806 ] && echo yes || echo NO)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cd $root/openpi && find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
setsid nohup bash cluster_v6/task1/queue_task1A_hgx1.sh >> $logs/queue_task1A_hgx1.out 2>&1 < /dev/null & disown
sleep 30
echo "--- re-armed:"; tail -2 $logs/queue_task1_hgx1.log; tail -1 $logs/train_v6_task1A_20260908_r1_status.log
