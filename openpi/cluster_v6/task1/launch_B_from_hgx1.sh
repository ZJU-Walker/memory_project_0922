#!/usr/bin/env bash
# Run ON iris-hgx-1: stop stage A (job 17315830; never the train_hs.py keep-alive), protect ckpt <step> as keep_<step>,
# launch stage B from it (pi05_yam_mem_v6_task1B: own writes, save 250 / keep 500) on the 4 H100.
#   cluster_v6/task1/launch_B_from_hgx1.sh <A step>
# Script file on purpose (pgrep self-match). Log: v6/logs/gate_task1_hgx2.log (shared with the hgx-2 gate).
export HOME=/iris/u/kewalk
step="$1"
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6
acfg=pi05_yam_mem_v6_task1A; aexp=v6_task1A_20260908_r1; bcfg=pi05_yam_mem_v6_task1B; bexp=v6_task1B_20260908_r1
ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-1] $*" >> "$log"; }
pat_a="scripts/train.py pi05_yam_mem_v6_task1[A]|queue_task1A_hgx[1]|run_train_hgx1.sh pi05_yam_mem_v6_task1[A]"
say "stopping stage A for stage B from ckpt $step"
pids=$(pgrep -u kewalk -f "$pat_a"); [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 20
left=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[A]"); [ -n "$left" ] && { kill -KILL $left; sleep 10; }
until ! pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[A]" >/dev/null; do sleep 5; done
sleep 30
say "stage A stopped; keep-alive 3743806 $([ -d /proc/3743806 ] && echo alive || echo GONE); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
[ -d "$ckroot/$acfg/$aexp/keep_$step" ] || cp -r "$ckroot/$acfg/$aexp/$step" "$ckroot/$acfg/$aexp/keep_$step"
[ -e "$ckroot/$acfg/$aexp/keep_$step/params" ] && say "ckpt $step protected as keep_$step" || { say "keep_$step COPY FAILED (stop)"; exit 1; }
export OPENPI_V6_TASK1_A_PARAMS="v6/checkpoints/$acfg/$aexp/keep_$step/params"
cd $root/openpi && find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
say "launching stage B ($bexp) from $OPENPI_V6_TASK1_A_PARAMS"
JOB=17315830 GPUS=4 BATCH=8 setsid nohup bash $cv6/run_train_hgx1.sh $bcfg $bexp > $logs/launch_B.out 2>&1 < /dev/null & disown
sleep 5; tail -1 $logs/train_${bexp}_status.log >> "$log"
