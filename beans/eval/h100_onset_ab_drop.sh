#!/bin/bash
# after the trend runs: episode 3, step 1, checkpoint 2000 with prefill rows dropped -- (a) no light_on note (bank = wait, off 1),
# (b) no wait note (bank = on 1, off 1), (c) both dropped (bank = off 1 only); then (a) at 1750.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h100_onset_ab_tren[d].sh|h100_onset_ab_step[s].sh|h100_onset_ab_se[t].sh" >/dev/null; do sleep 20; done
run() { local st=$1 tag=$2; shift 2; STEP=$st EXTRA="--step 1 $*" bash beans/eval/h100_onset_ab.sh 3; mv beans/eval/onset_ab_ep03_$st.log beans/eval/onset_ab_ep03_${st}_step1_$tag.log; }
run 2000 noon --drop-note-prefix light_on
run 2000 nowait --drop-note-prefix wait
run 2000 offonly --drop-note-prefix light_on --drop-note-prefix wait
run 1750 noon --drop-note-prefix light_on
echo "onset A/B drop set done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
