#!/bin/bash
# H100 card 0 (job 17489557): row-count test under blank images (onset window, step 1) -- bank reduced to fewer rows.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h100_battery_train_spli[t].sh|h100_onset_ab_input[s].sh" >/dev/null; do sleep 20; done
run() { EP=$1; TAG=$2; EXTRA=$3; STEP=2000 EXTRA="$EXTRA" bash beans/eval/h100_onset_ab.sh $EP; mv beans/eval/onset_ab_ep$(printf %02d $EP)_2000.log beans/eval/onset_ab_ep$(printf %02d $EP)_2000$TAG.log; }
run 3 _step1_blankimg_offonly "--step 1 --blank-images --drop-note-prefix light_on --drop-note-prefix wait_for"
run 3 _step1_blankimg_noon "--step 1 --blank-images --drop-note-prefix light_on"
run 0 _step1_blankimg_noon "--step 1 --blank-images --drop-note-prefix light_on"
run 3 _win41_step8_blankimg "--frame 41 --step 8 --blank-images"
echo "H100 seqE done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
