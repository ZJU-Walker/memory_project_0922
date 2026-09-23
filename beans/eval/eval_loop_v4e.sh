#!/bin/bash
# H100 card 0 (job 17489557): after every new v4e checkpoint, the true-onset A/B (prefill notes only, step 1 after the onset)
# for training episodes 3 and 8 (x=1) and 2 (x=2); one summary line per checkpoint in beans/eval/v4e_onset_summary.log.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
RUN=beans/checkpoints/pi05_yam_beans0922_v4e/beans0922_v4e; OUT=beans/eval/v4e_onset_summary.log; touch $OUT
while true; do
  for d in $(ls $RUN 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    [ "$d" -le 500 ] && continue; grep -q "^ckpt $d " $OUT && continue
    [ -d "$RUN/$d/params" ] || continue; sleep 60  # let the save finish
    line="ckpt $d"
    for ep in 3 8 2; do
      STEP=$d TAG=_step1 EXTRA="--step 1" bash beans/eval/h100_onset_ab_cfg.sh $ep
      f=beans/eval/onset_ab_v4e_ep$(printf %02d $ep)_${d}_step1.log
      p=$(grep -m1 'count=1: -log p sum' $f 2>/dev/null | sed 's/.*= //'); g=$(grep -m1 'greedy decode' $f 2>/dev/null | sed "s/.*decode: '//; s/' lowest.*//" | cut -c1-44)
      line="$line | ep$ep p(1,2,3)=${p:-NA} greedy='${g:-NA}'"
    done
    echo "$line $(date +%H:%M)" >> $OUT
  done
  sleep 120
done
