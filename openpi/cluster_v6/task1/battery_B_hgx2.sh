#!/usr/bin/env bash
# Run ON iris-hgx-2 (job 17329416): development battery on every stage-B checkpoint of v6_task1B_20260908_r1 as it
# appears (250, 500, ..., 2000; the non-multiples of 500 are deleted by the checkpoint manager at the next save, so each
# battery starts right after its checkpoint is finalized). Oracle + self writes in parallel on the H200.
# Log: v6/logs/gate_task1_hgx2.log. Script file on purpose (pgrep self-match).
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; diag=$root/v6/diagnostics
bcfg=pi05_yam_mem_v6_task1B; bexp=v6_task1B_20260908_r1; ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
finalized() { [ -e "$ckroot/$bcfg/$bexp/$1/params" ] && grep -q "\[step=$1\] CheckpointManager Save Finalize is done on all hosts" "$logs/train_$bexp.log" 2>/dev/null; }
say "stage-B battery loop armed on $(hostname) (B checkpoints 250..2000 as they appear)"
for step in 250 500 750 1000 1250 1500 1750 2000; do
  until finalized $step; do
    if [ "$(tail -1 $logs/train_${bexp}_status.log 2>/dev/null | cut -c1-5)" = "exit=" ] && ! finalized $step; then
      say "stage B exited before ckpt $step ($(tail -1 $logs/train_${bexp}_status.log)); stopping"; step=""; break
    fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "B ckpt $step finalized -> battery on the H200 (oracle + self)"
  MODES=oracle JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh $bcfg $bexp $step &
  MODES=self JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh $bcfg $bexp $step &
  wait
  (cd $root/openpi && .venv/bin/python scripts/task1_battery_verdict.py "$diag/videos_${bexp}_${step}") >> "$log"
done
say "stage-B battery loop done"
