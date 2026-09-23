#!/bin/bash
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
export GPU=0
STEP=2000 TAG=_step1_normal EXTRA="--step 1" bash beans/eval/h200_onset_ab.sh 0
STEP=2000 TAG=_step1_normal EXTRA="--step 1" bash beans/eval/h200_onset_ab.sh 2
STEP=2000 TAG=_step1_zerostate EXTRA="--step 1 --zero-state" bash beans/eval/h200_onset_ab.sh 3
echo "H200 seqA done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
