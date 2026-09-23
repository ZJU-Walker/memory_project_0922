#!/bin/bash
# stop the v4c training on hgx-2 (launcher chain + srun front-end + train.py step); never touches train_hs.py or GPUs 2,3.
LIST=$(ps -eo pid,ppid,args --no-headers | grep -E "train_beans0922_v4[.]sh|EXP=beans0922_v4[c] |beans0922_v4[c] --exp-name" | grep -v "grep -E")
echo "$LIST" | cut -c1-120
PIDS=$(echo "$LIST" | awk '{print $1}' | tr '\n' ' ')
[ -n "$PIDS" ] && kill -TERM $PIDS 2>/dev/null; sleep 20
for i in 1 2 3 4; do P=$(ps -eo pid,args --no-headers | grep -E "train[.]py pi05_yam_beans0922_v4c " | awk '{print $1}'); [ -z "$P" ] && break; kill -TERM $P 2>/dev/null; sleep 15; done
P=$(ps -eo pid,args --no-headers | grep -E "train[.]py pi05_yam_beans0922_v4c " | awk '{print $1}'); [ -n "$P" ] && { kill -KILL $P; sleep 5; }
echo "train remnants: $(ps -eo args --no-headers | grep -E "train[.]py pi05_yam_beans0922_v4c " | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
