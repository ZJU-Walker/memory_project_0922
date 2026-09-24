#!/bin/bash
# H100 card 0 (job 17489557): after every new v4e checkpoint (1) the true-onset A/B for training episodes 3, 8 (x=1) and 2 (x=2),
# (2) own-note rollouts of dev episodes 29 and 73 (x=1) with a copy-stability digest (user 17:05: the sentence flickered after
# the go light in v4b/v4c). One line per checkpoint in beans/eval/v4e_onset_summary.log.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
RUN=beans/checkpoints/pi05_yam_beans0922_v4e/beans0922_v4e; OUT=beans/eval/v4e_onset_summary.log; touch $OUT
digest() {  # $1 = rollout json -> "go=<own go count hist> changes=<sentence changes after go> kofx=<hist>"
python3 - "$1" <<'PY'
import json, re, sys, collections
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("NA"); sys.exit()
seq=[(r["step"], r["pred"], r["gt_now"]) for r in d["records"]]
go=next((s for s,_,g in seq if g.startswith("yellow go")), None)
ch=0; prev=None
for s,p,g in seq:
    if go is None or s<go: continue
    if p[:60]!=prev: ch+=1; prev=p[:60]
cnt=collections.Counter(m.group(1) for s,p,g in seq if go is not None and s>=go for m in [re.search(r"scoop (\d) time",p)] if m)
kofx=collections.Counter(m.group(1) for s,p,g in seq for m in [re.search(r"scoop (\d of \d)",p)] if m)
print(f"go={dict(cnt)} changes={ch} kofx={dict(kofx)}")
PY
}
while true; do
  for d in $(ls $RUN 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    [ "$d" -le 500 ] && continue; grep -q "^ckpt $d " $OUT && continue
    [ -d "$RUN/$d/params" ] || continue; sleep 60
    line="ckpt $d"
    for ep in 3 8 2; do
      STEP=$d TAG=_step1 EXTRA="--step 1" bash beans/eval/h100_onset_ab_cfg.sh $ep
      f=beans/eval/onset_ab_v4e_ep$(printf %02d $ep)_${d}_step1.log
      p=$(grep -m1 'count=1: -log p sum' $f 2>/dev/null | sed 's/.*= //'); g=$(grep -m1 'greedy decode' $f 2>/dev/null | sed "s/.*decode: '//; s/' lowest.*//" | cut -c1-44)
      line="$line | ep$ep p(1,2,3)=${p:-NA} greedy='${g:-NA}'"
    done
    JOB=17489557 GPU=0 GRES=2 CPUS=8 MODES=self EPISODES="29 73" bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v4e beans0922_v4e "$d" >/dev/null 2>&1
    for ep in 29 73; do line="$line | roll ep$ep $(digest beans/eval/videos_beans0922_v4e_${d}/ep${ep}_self.json)"; done
    echo "$line $(date +%H:%M)" >> $OUT
  done
  sleep 120
done
