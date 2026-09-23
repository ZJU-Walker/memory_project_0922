#!/bin/bash
# after the step runs: onset A/B at step 1 (yellow visible) for training ep 3 at v4c checkpoints 1000, 1500, 2000, and ep 8 at 2000.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
while pgrep -u "$USER" -f "h100_onset_ab_se[t].sh|h100_onset_ab_step[s].sh" >/dev/null; do sleep 20; done
for spec in "3 1000" "3 1500" "3 2000" "8 2000"; do
  set -- $spec; ep=$1; st=$2
  [ -d beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c/$st/params ] || { echo "no ckpt $st" >> beans/eval/battery_in_job_status.log; continue; }
  STEP=$st EXTRA="--step 1" bash beans/eval/h100_onset_ab.sh $ep
  mv beans/eval/onset_ab_ep$(printf %02d $ep)_$st.log beans/eval/onset_ab_ep$(printf %02d $ep)_${st}_step1.log
done
echo "onset A/B trend done $(date +%H:%M)" >> beans/eval/battery_in_job_status.log
