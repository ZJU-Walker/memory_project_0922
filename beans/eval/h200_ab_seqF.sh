#!/bin/bash
# GPU 0 after seqC: image-only (bank dropped) and bank-only (blank images) at the onset window, step 1, for x=3 (ep0) and x=2 (ep2).
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h200_ab_seq[C].sh" >/dev/null; do sleep 20; done
export GPU=0
STEP=2000 TAG=_step1_nonotes EXTRA="--step 1 --drop-note-prefix ALL" bash beans/eval/h200_onset_ab.sh 0
STEP=2000 TAG=_step1_nonotes EXTRA="--step 1 --drop-note-prefix ALL" bash beans/eval/h200_onset_ab.sh 2
STEP=2000 TAG=_step1_blankimg EXTRA="--step 1 --blank-images" bash beans/eval/h200_onset_ab.sh 2
echo "H200 seqF done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
