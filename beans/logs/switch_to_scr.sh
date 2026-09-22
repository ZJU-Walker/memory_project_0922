#!/usr/bin/env bash
# 2026-09-22 04:30: once base ckpt 2500 is finalized and the node-local dataset copy is complete, restart the chain on the local copy
# (the launcher resumes from the numeric checkpoint). ON iris-hgx-1.
ROOT=/iris/u/kewalk/memory_project_beans0922; L=$ROOT/beans/logs/train_beans0922_base.log; SCR=/scr/kewalk/beans0922/bean_scoop_0905_v5
log() { echo "[$(date +%m/%d\ %H:%M:%S)] switch: $*" | tee -a $ROOT/beans/logs/train_beans0922_status.log; }
n=0; until { tail -c 200000 $L | grep -a -q "step=2500.*Save Finalize is done" && [ "$(find $SCR -type f | wc -l)" = 96 ] && ! pgrep -u kewalk -f 'rsync -a /iris/u/kewalk/memory_project_v5' >/dev/null; } || [ $n -ge 240 ]; do sleep 15; n=$((n+1)); done
[ -d $ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/2500 ] || { log "no finalized 2500 after the wait; not switching"; exit 1; }
log "ckpt 2500 finalized, local copy complete ($(du -sh $SCR | cut -f1)); restarting the chain on $SCR"
bash $ROOT/beans/logs/stop_all_beans0922.sh >/dev/null 2>&1; sleep 5
cd $ROOT && export JOB=17489557 GPUS=0,1 WAIT_ROUNDS=2880 FREE_STREAK=2 OPENPI_BEANS_DATASET_ROOT=$SCR && bash beans/logs/beans0922_ctl.sh start
