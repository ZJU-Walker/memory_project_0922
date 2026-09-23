#!/bin/bash
# h100_eval_chain.sh STEP : on H100 card 0 of job 17489557 (user 09-23 12:58 "use this card"), run the six own-note videos of
# v4c/STEP, then the count battery. Launch from hgx-1: setsid nohup bash beans/eval/h100_eval_chain.sh STEP > log 2>&1 < /dev/null &
set -u
STEP=${1:?step}; cd /iris/u/kewalk/memory_project_beans0922 || exit 2
export JOB=${JOB:-17489557} GPU=${GPU:-0} GRES=${GRES:-2} CPUS=${CPUS:-8}
MODES=self EPISODES="${EPISODES:-29 73 25 72 64 59}" bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v4c beans0922_v4c "$STEP"
[ "${SKIP_BATTERY:-0}" = 1 ] || bash beans/eval/run_battery_in_job.sh pi05_yam_beans0922_v4c beans0922_v4c "$STEP"
