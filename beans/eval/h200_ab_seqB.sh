#!/bin/bash
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
export GPU=1
STEP=2000 TAG=_step1_blankimg EXTRA="--step 1 --blank-images" bash beans/eval/h200_onset_ab.sh 0
STEP=2000 TAG=_step1_blankimg EXTRA="--step 1 --blank-images" bash beans/eval/h200_onset_ab.sh 8
STEP=2000 TAG=_step1_blankimg_zerostate EXTRA="--step 1 --blank-images --zero-state" bash beans/eval/h200_onset_ab.sh 3
echo "H200 seqB done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
