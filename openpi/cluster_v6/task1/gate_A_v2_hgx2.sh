#!/usr/bin/env bash
# v2 gate, run ON iris-hgx-2 (job 17329416, H200). For each stage-A checkpoint 500, 750, ... (250 already tested): wait
# until finalized, battery in three write modes in parallel (oracle, oracle_evidence, self), verdict from
# scripts/task1_battery_verdict.py. PASS = RECALL under oracle_evidence (label notes, own closing sentence) right on
# >= 5/6 development episodes at the first closing step -> ssh hgx-1 launch_B_from_hgx1.sh <step>, then the three-mode
# battery on every B checkpoint (250, 500, ...). Log: v6/logs/gate_task1_hgx2.log. Script file on purpose.
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; diag=$root/v6/diagnostics
acfg=pi05_yam_mem_v6_task1A; aexp=v6_task1A_20260908_r1; bcfg=pi05_yam_mem_v6_task1B; bexp=v6_task1B_20260908_r1
ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
SSH="ssh -i /iris/u/kewalk/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
finalized() { [ -e "$ckroot/$1/$2/$3/params" ] && grep -q "\[step=$3\] CheckpointManager Save Finalize is done on all hosts" "$logs/train_$2.log" 2>/dev/null; }
battery() {  # cfg exp step -> verdict text
  for m in oracle oracle_evidence self; do MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh "$1" "$2" "$3" & done
  wait
  (cd $root/openpi && .venv/bin/python scripts/task1_battery_verdict.py "$diag/videos_$2_$3")
}
exited() { [ "$(tail -1 $logs/train_$1_status.log 2>/dev/null | cut -c1-5)" = "exit=" ]; }
say "gate v2 armed on $(hostname): PASS = oracle_evidence recall >= 5/6; A checkpoints 500.. as they appear"
passed=""
for step in 500 750 1000 1250 1500 1750 2000; do
  until finalized $acfg $aexp $step; do
    if exited $aexp && ! finalized $acfg $aexp $step; then say "stage A exited before ckpt $step ($(tail -1 $logs/train_${aexp}_status.log)); A loop ends"; step=""; break; fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "A ckpt $step finalized -> battery on the H200 (oracle, oracle_evidence, self)"
  verdict=$(battery $acfg $aexp $step); echo "$verdict" >> "$log"
  n=$(echo "$verdict" | grep -o "RECALL (first closing step) — oracle [0-9]/6, oracle_evidence [0-9]" | grep -o "[0-9]$")
  if [ -n "$n" ] && [ "$n" -ge 5 ]; then passed=$step; break; fi
  say "A ckpt $step: oracle_evidence recall ${n:-?}/6 < 5 -> waiting for the next checkpoint"
done
if [ -z "$passed" ]; then say "no stage-A checkpoint passed the recall gate; gate script done"; exit 0; fi
say "A ckpt $passed PASSED the recall gate -> stage B from it (hgx-1)"
$SSH iris-hgx-1 "bash $cv6/task1/launch_B_from_hgx1.sh $passed" >> $logs/gate_ssh.out 2>&1 || say "ssh to hgx-1 FAILED (launch B by hand: launch_B_from_hgx1.sh $passed)"
for step in 250 500 750 1000 1250 1500 1750 2000; do
  until finalized $bcfg $bexp $step; do
    if exited $bexp && ! finalized $bcfg $bexp $step; then say "stage B exited before ckpt $step ($(tail -1 $logs/train_${bexp}_status.log)); B loop ends"; step=""; break; fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "B ckpt $step finalized -> battery on the H200 (oracle, oracle_evidence, self)"
  battery $bcfg $bexp $step >> "$log"
done
say "gate v2 done"
