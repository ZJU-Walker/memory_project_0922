#!/usr/bin/env bash
# Run ON iris-hgx-1: stop stage B (job 17315830; never the train_hs.py keep-alive) and RESUME stage A from its last
# numeric checkpoint (run_train_hgx1.sh picks --resume). Decision 2026-09-08 23:40: the clean recall test on A ckpt 250
# (oracle_evidence) is 2/6 — the lookup of an older note by object is not learned yet, and B's own (wrong) notes would
# corrupt the closing-step signal that teaches it. Script file on purpose (pgrep self-match).
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6
acfg=pi05_yam_mem_v6_task1A; aexp=v6_task1A_20260908_r1; log=$logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-1] $*" >> "$log"; }
pat_b="scripts/train.py pi05_yam_mem_v6_task1[B]|run_train_hgx1.sh pi05_yam_mem_v6_task1[B]"
say "stopping stage B (recall not learned at A-250) and resuming stage A from 250"
pids=$(pgrep -u kewalk -f "$pat_b"); [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 20
left=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[B]"); [ -n "$left" ] && { kill -KILL $left; sleep 10; }
until ! pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[B]" >/dev/null; do sleep 5; done
sleep 30
say "stage B stopped; keep-alive 3743806 $([ -d /proc/3743806 ] && echo alive || echo GONE); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
cd $root/openpi && find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
JOB=17315830 GPUS=4 BATCH=8 setsid nohup bash $cv6/run_train_hgx1.sh $acfg $aexp > $logs/resume_A.out 2>&1 < /dev/null & disown
sleep 5; tail -1 $logs/train_${aexp}_status.log >> "$log"
