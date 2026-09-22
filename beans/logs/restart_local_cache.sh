#!/usr/bin/env bash
# 2026-09-22 05:45: restart the chain (resume from the last numeric checkpoint) with BOTH the dataset and the HF datasets arrow cache
# on the node-local disk; the arrow cache on NFS was memory-mapped by the loader workers (page-fault waits, GPUs idle 80 % of the time).
ROOT=/iris/u/kewalk/memory_project_beans0922; SCR=/scr/kewalk/beans0922
log() { echo "[$(date +%m/%d\ %H:%M:%S)] restart: $*" | tee -a $ROOT/beans/logs/train_beans0922_status.log; }
bash $ROOT/beans/logs/stop_all_beans0922.sh >/dev/null 2>&1; sleep 5
mkdir -p $SCR/hf_datasets
log "chain restarted with OPENPI_BEANS_DATASET_ROOT=$SCR/bean_scoop_0905_v5 HF_DATASETS_CACHE=$SCR/hf_datasets (resume from $(ls $ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base | grep -E '^[0-9]+$' | sort -n | tail -1))"
cd $ROOT && export JOB=17489557 GPUS=0,1 WAIT_ROUNDS=2880 FREE_STREAK=2 OPENPI_BEANS_DATASET_ROOT=$SCR/bean_scoop_0905_v5 HF_DATASETS_CACHE=$SCR/hf_datasets && bash beans/logs/beans0922_ctl.sh start
