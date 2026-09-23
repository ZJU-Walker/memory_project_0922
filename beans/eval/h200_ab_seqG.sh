#!/bin/bash
# GPU 1 after seqD: the row-count test under blank images (moved here from the H100, whose battery blocks it).
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h200_ab_seq[D].sh" >/dev/null; do sleep 20; done
export GPU=1
STEP=2000 TAG=_step1_blankimg_offonly EXTRA="--step 1 --blank-images --drop-note-prefix light_on --drop-note-prefix wait_for" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_step1_blankimg_noon EXTRA="--step 1 --blank-images --drop-note-prefix light_on" bash beans/eval/h200_onset_ab.sh 3
STEP=2000 TAG=_step1_blankimg_noon EXTRA="--step 1 --blank-images --drop-note-prefix light_on" bash beans/eval/h200_onset_ab.sh 0
STEP=2000 TAG=_win41_step8_blankimg EXTRA="--frame 41 --step 8 --blank-images" bash beans/eval/h200_onset_ab.sh 3
echo "H200 seqG done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
