#!/bin/bash
# v4e (onset-focused sampling + copy-tick weight 0.2, write rule = v4c) from the clean v4/500 checkpoint on the 2 x H200
# (hgx-2 job 17425063 GPUs 0,1; user 16:34 "从500开始吧，干净一点，2h200"). Idempotent: seeds the run dir once, then launches
# beans/logs/train_beans0922_v4.sh in MODE=mem (it resumes from the highest numeric step dir = 500).
set -u
R=/iris/u/kewalk/memory_project_beans0922; C=$R/beans/checkpoints; RUN=$C/pi05_yam_beans0922_v4e/beans0922_v4e
if [ ! -d "$RUN/500" ]; then
  mkdir -p "$RUN"; cp -al "$C/archive/v4_500" "$RUN/500" || { echo "seed failed"; exit 2; }
  echo "seeded $RUN/500 from archive/v4_500 (hardlinks) $(date +%H:%M)"
fi
ls "$RUN/500" | tr '\n' ' '; echo
cd "$R" || exit 2
JOB=17425063 GPUS=0,1 MODE=mem CFG=pi05_yam_beans0922_v4e EXP=beans0922_v4e BATCH=8 BATCH_FALLBACK=4 \
  setsid nohup bash "$R/beans/logs/train_beans0922_v4.sh" > "$R/beans/logs/beans0922_v4e.launch.log" 2>&1 < /dev/null &
sleep 3; echo "launched: $(pgrep -u "$USER" -af 'train_beans0922_v4[.]sh' | cut -c1-80)"
