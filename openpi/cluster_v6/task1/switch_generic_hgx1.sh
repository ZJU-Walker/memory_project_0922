#!/usr/bin/env bash
# Run ON iris-hgx-1: stop the task1 training that currently holds job 17315830 (any pi05_yam_mem_v6_task1* run; never
# the train_hs.py keep-alive) and launch another config fresh/resumed.
#   [JOB=17315830] cluster_v6/task1/switch_generic_hgx1.sh <config> <exp> [note]
export HOME=/iris/u/kewalk
cfg="$1"; exp="$2"; note="${3:-}"
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; log=$logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-1] $*" >> "$log"; }
pat="scripts/train.py pi05_yam_mem_v6_task[1]|run_train_hgx1.sh pi05_yam_mem_v6_task[1]|queue_task1A_hgx[1]"
say "switch -> $cfg / $exp ${note:+($note)}: stopping the current task1 training"
pids=$(pgrep -u kewalk -f "$pat"); [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 20
left=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task[1]"); [ -n "$left" ] && { kill -KILL $left; sleep 10; }
until ! pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task[1]" >/dev/null; do sleep 5; done
sleep 30
say "stopped; keep-alive 3743806 $([ -d /proc/3743806 ] && echo alive || echo GONE); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
for f in $logs/train_v6_task1*_status.log; do  # the killed wrapper cannot log its own exit: write it for the gate's `exited` check
  [ "$(tail -1 "$f" | cut -c1-6)" = "launch" ] && echo "exit=143 $(date +%H:%M) stopped by $(basename "$0")" >> "$f"
done
cd $root/openpi && find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
JOB=${JOB:-17315830} GPUS=4 BATCH=8 setsid nohup bash $cv6/run_train_hgx1.sh "$cfg" "$exp" > $logs/launch_${exp}.out 2>&1 < /dev/null & disown
sleep 5; tail -1 $logs/train_${exp}_status.log >> "$log"
