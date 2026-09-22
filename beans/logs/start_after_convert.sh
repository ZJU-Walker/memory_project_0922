#!/usr/bin/env bash
# 2026-09-22: wait for the dataset rebuild (convert_beans0905.out ends with "convert exit=0"), run the 2-update base smoke, then
# start the chain (base 10k -> memory 5k) detached on job $JOB. ON iris-hgx-1.
set -u
ROOT=/iris/u/kewalk/memory_project_beans0922; OUT=/iris/u/kewalk/memory_project_0920/beans/logs/convert_beans0905.out
status=$ROOT/beans/logs/train_beans0922_status.log; log() { echo "[$(date +%m/%d\ %H:%M:%S)] starter: $*" | tee -a "$status"; }
n=0; until tr -d '\000' < $OUT 2>/dev/null | grep -q "convert exit=" || [ $n -ge 360 ]; do sleep 20; n=$((n+1)); done
line=$(tr -d '\000' < $OUT | grep "convert exit=" | tail -1); log "conversion: ${line:-timed out}"
echo "$line" | grep -q "exit=0" || { log "conversion did not succeed; not starting"; exit 1; }
ds=$ROOT/v5/data/lerobot/yam/bean_scoop_0905_v5; eps=$(wc -l < $ds/meta/episodes.jsonl 2>/dev/null); log "dataset episodes: ${eps:-0}"
[ "${eps:-0}" = 89 ] || { log "expected 89 episodes; not starting"; exit 1; }
export JOB=${JOB:-17489557} GPUS=0,1
log "smoke_base"; MODE=smoke_base bash $ROOT/beans/logs/train_beans0922.sh || { log "smoke_base failed; not starting the chain"; exit 1; }
log "starting the chain"; bash $ROOT/beans/logs/beans0922_ctl.sh start
