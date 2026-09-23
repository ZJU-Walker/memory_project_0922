#!/bin/bash
# H100 card 0 after the window dump: the A/B in the battery's framing for ep3 -- window start 2 ticks before the go
# onset (frame 71, onset = step 2), bank as is and bank emptied (the battery's "blank" condition).
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h100_battery_train_dum[p].sh" >/dev/null; do sleep 15; done
run() { EP=$1; TAG=$2; EXTRA=$3; STEP=2000 EXTRA="$EXTRA" bash beans/eval/h100_onset_ab.sh $EP; mv beans/eval/onset_ab_ep$(printf %02d $EP)_2000.log beans/eval/onset_ab_ep$(printf %02d $EP)_2000$TAG.log; }
run 3 _win71_step2 "--frame 71 --step 2"
run 3 _win71_step2_nonotes "--frame 71 --step 2 --drop-note-prefix ALL"
echo "H100 seqH done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
