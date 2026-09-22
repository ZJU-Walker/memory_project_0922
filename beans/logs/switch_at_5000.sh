#!/usr/bin/env bash
# 2026-09-22 06:10: at base ckpt 5000 (finalized), restart the chain so the loader uses the node-local arrow cache via the in-tree
# symlink v35/cache/huggingface/datasets -> /scr/kewalk/beans0922/hf_datasets (train.py pins HF_DATASETS_CACHE to the in-tree path,
# so exporting it did nothing; the workers were memory-mapping the arrow files over NFS). Resume from 5000. ON iris-hgx-1.
ROOT=/iris/u/kewalk/memory_project_beans0922; L=$ROOT/beans/logs/train_beans0922_base.log; SCR=/scr/kewalk/beans0922
log() { echo "[$(date +%m/%d\ %H:%M:%S)] switch5000: $*" | tee -a $ROOT/beans/logs/train_beans0922_status.log; }
n=0; until tail -c 300000 $L | grep -a -q "step=5000.*Save Finalize is done" || [ $n -ge 720 ]; do sleep 15; n=$((n+1)); done
[ -d $ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/5000 ] || { log "no finalized 5000 after the wait; not switching"; exit 1; }
[ -L $ROOT/v35/cache/huggingface/datasets ] || { log "in-tree datasets cache is not the /scr symlink; not switching"; exit 1; }
log "ckpt 5000 finalized; restarting the chain (dataset $SCR/bean_scoop_0905_v5, arrow cache via the in-tree symlink)"
bash $ROOT/beans/logs/stop_all_beans0922.sh >/dev/null 2>&1; sleep 5
cd $ROOT && export JOB=17489557 GPUS=0,1 WAIT_ROUNDS=2880 FREE_STREAK=2 OPENPI_BEANS_DATASET_ROOT=$SCR/bean_scoop_0905_v5 && unset HF_DATASETS_CACHE && bash beans/logs/beans0922_ctl.sh start
