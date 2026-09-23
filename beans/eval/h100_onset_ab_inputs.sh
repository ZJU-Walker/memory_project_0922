#!/bin/bash
# after the drop set: input ablations at v4c/2000, episode 3 -- (a) step 1, no notes at all; (b) step 1, images blanked;
# (c) step 8 (mid go), no notes; (d) step 8, normal bank; (e) step 1, no notes AND images blanked.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h100_onset_ab_dro[p].sh" >/dev/null; do sleep 20; done
run() { local tag=$1; shift; STEP=2000 EXTRA="$*" bash beans/eval/h100_onset_ab.sh 3; mv beans/eval/onset_ab_ep03_2000.log beans/eval/onset_ab_ep03_2000_$tag.log; }
run step1_nonotes --step 1 --drop-note-prefix ALL
run step1_blankimg --step 1 --blank-images
run step8_nonotes --step 8 --drop-note-prefix ALL
run step8_normal --step 8
run step1_nonotes_blankimg --step 1 --drop-note-prefix ALL --blank-images
echo "onset A/B input set done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
