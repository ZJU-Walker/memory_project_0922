#!/usr/bin/env bash
# Run ON iris-hgx-2 (job 17329416, the H200; user 2026-09-08 21:12: "for testing A you can use 17329416 so you can keep
# A training"). Stage-A gate + stage-B batteries, all on the H200 while training keeps the 4 H100 of job 17315830:
#   for each stage-A checkpoint 250, 500, ...: wait until it is finalized, run the development battery (oracle writes
#   and self writes in parallel on the H200), verdict = oracle first decision correct on >= 5/6 episodes;
#   PASS -> ssh iris-hgx-1 launch_B_from_hgx1.sh <step> (stops A, launches B from keep_<step>), then battery on every
#   kept B checkpoint (500, 1000, 1500, 2000) as it appears; FAIL -> next A checkpoint.
# Log: v6/logs/gate_task1_hgx2.log. Script file on purpose (pgrep self-match).
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; diag=$root/v6/diagnostics
acfg=pi05_yam_mem_v6_task1A; aexp=v6_task1A_20260908_r1; bcfg=pi05_yam_mem_v6_task1B; bexp=v6_task1B_20260908_r1
ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
SSH="ssh -i /iris/u/kewalk/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
finalized() { [ -e "$ckroot/$1/$2/$3/params" ] && grep -q "\[step=$3\] CheckpointManager Save Finalize is done on all hosts" "$logs/train_$2.log" 2>/dev/null; }
battery() {  # cfg exp step -> verdict text
  MODES=oracle JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh "$1" "$2" "$3" &
  MODES=self JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh "$1" "$2" "$3" &
  wait
  (cd $root/openpi && .venv/bin/python scripts/task1_battery_verdict.py "$diag/videos_$2_$3")
}
say "gate armed on $(hostname), job 17329416; A checkpoints tested as they appear"
passed=""
for step in 250 500 750 1000 1250 1500 1750 2000; do
  until finalized $acfg $aexp $step; do
    if grep -q "^exit=" $logs/train_${aexp}_status.log 2>/dev/null && ! pgrep -u kewalk -f "run_train_hgx1.sh pi05_yam_mem_v6_task1[A]" >/dev/null 2>&1 \
       && [ "$(tail -1 $logs/train_${aexp}_status.log | cut -c1-5)" = "exit=" ] && ! finalized $acfg $aexp $step; then
      # A exited (or was stopped) before this checkpoint: nothing more to test on the A side
      say "stage A exited before ckpt $step; stopping the A loop"; step=""; break
    fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "A ckpt $step finalized -> battery on the H200 (oracle + self)"
  verdict=$(battery $acfg $aexp $step); echo "$verdict" >> "$log"
  if echo "$verdict" | grep -q "^GATE PASS"; then passed=$step; break; fi
  say "A ckpt $step did not pass; waiting for the next one"
done
if [ -z "$passed" ]; then say "no stage-A checkpoint passed the gate; gate script done"; exit 0; fi
say "A ckpt $passed PASSED -> stage B from it (hgx-1)"
$SSH iris-hgx-1 "bash $cv6/task1/launch_B_from_hgx1.sh $passed" >> $logs/gate_ssh.out 2>&1 || say "ssh to hgx-1 FAILED (launch B by hand: cluster_v6/task1/launch_B_from_hgx1.sh $passed)"
for step in 500 1000 1500 2000; do
  until finalized $bcfg $bexp $step; do
    if [ "$(tail -1 $logs/train_${bexp}_status.log 2>/dev/null | cut -c1-5)" = "exit=" ] && ! finalized $bcfg $bexp $step; then
      say "stage B exited before ckpt $step ($(tail -1 $logs/train_${bexp}_status.log)); stopping"; step=""; break
    fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "B ckpt $step finalized -> battery on the H200 (oracle + self)"
  verdict=$(battery $bcfg $bexp $step); echo "$verdict" >> "$log"
done
say "gate script done"
