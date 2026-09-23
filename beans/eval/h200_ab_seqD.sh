#!/bin/bash
# GPU 1 after seqB: training-like windows -- start 8/16 ticks BEFORE the go onset so the earlier ticks are in-window
# (teacher forced), the onset is step 8/16.  ep3 go=81 (stride 5): frame 41 -> step 8, frame 1 -> step 16.  ep8 go=89: frame 49 -> step 8.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h200_ab_seq[B].sh" >/dev/null; do sleep 20; done
export GPU=1
STEP=2000 TAG=_win41_step8 EXTRA="--frame 41 --step 8" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_win1_step16 EXTRA="--frame 1 --step 16" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_win49_step8 EXTRA="--frame 49 --step 8" bash beans/eval/h200_onset_ab.sh 8
STEP=2000 TAG=_win1_step16_blankimg EXTRA="--frame 1 --step 16 --blank-images" bash beans/eval/h200_onset_ab.sh 3
echo "H200 seqD done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
