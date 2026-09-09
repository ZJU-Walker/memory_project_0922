#!/usr/bin/env bash
# v6 task1 stage A queue on iris-hgx-1 (job 17315830): wait for the task1 norm stats, train pi05_yam_mem_v6_task1A
# (exp v6_task1A_20260908_r1, 4xH100, batch 8, 2000 updates, checkpoints every 250), then the development battery
# (self + oracle writes) on GPU 0 for every kept checkpoint. Log: v6/logs/queue_task1_hgx1.log.
#   setsid nohup bash cluster_v6/task1/queue_task1A_hgx1.sh > v6/logs/queue_task1A_hgx1.out 2>&1 &
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6
cfg=pi05_yam_mem_v6_task1A; exp=v6_task1A_20260908_r1
stats=$root/v6/assets/pi05_yam_task1_0908_v6/yam/task1_find_0908_v6/norm_stats.json
echo "queue task1A armed on $(hostname) $(date '+%m/%d %H:%M') code=$(git -C $root rev-parse --short HEAD)" >> $logs/queue_task1_hgx1.log
until [ -e "$stats" ] && [ -e "$logs/norm_stats_task1v6.done" ]; do sleep 30; done
echo "norm stats present -> stage A starts $(date '+%m/%d %H:%M')" >> $logs/queue_task1_hgx1.log
JOB=17315830 GPUS=4 BATCH=8 bash $cv6/run_train_hgx1.sh $cfg $exp
code=$(grep "^exit=" $logs/train_${exp}_status.log | tail -1); echo "task1A r1 $code $(date '+%m/%d %H:%M')" >> $logs/queue_task1_hgx1.log
if ! echo "$code" | grep -q "exit=0"; then echo "task1A r1 failed (stop)" >> $logs/queue_task1_hgx1.log; exit 1; fi
for step in $(ls $root/v6/checkpoints/$cfg/$exp | grep -E '^[0-9]+$' | sort -n); do
  bash $cv6/task1/run_task1_evals_hgx1.sh $cfg $exp $step
  echo "battery $step: $(tail -1 $root/v6/diagnostics/videos_${exp}_${step}/status.log) $(date '+%m/%d %H:%M')" >> $logs/queue_task1_hgx1.log
done
echo "queue task1A done $(date '+%m/%d %H:%M')" >> $logs/queue_task1_hgx1.log
