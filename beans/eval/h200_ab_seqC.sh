#!/bin/bash
# after seqA (GPU 0): mid-go tick (step 8) ablations for ep 3 -- blank images; zero state; blank+zero+no notes; and ep0 (x=3) step 8 normal.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h200_ab_seq[A].sh" >/dev/null; do sleep 20; done
export GPU=0
STEP=2000 TAG=_step8_blankimg EXTRA="--step 8 --blank-images" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_step8_zerostate EXTRA="--step 8 --zero-state" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_step8_nothing EXTRA="--step 8 --blank-images --zero-state --drop-note-prefix ALL" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_step8_normal EXTRA="--step 8" bash beans/eval/h200_onset_ab.sh 0
echo "H200 seqC done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
