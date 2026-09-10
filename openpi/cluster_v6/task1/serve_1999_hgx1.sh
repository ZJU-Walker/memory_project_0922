#!/usr/bin/env bash
# User 2026-09-10 13:48: "dont run batteries anymore, directly serve 1999 on the h100". Run ON iris-hgx-1 (detached):
# waits for the B6 step-1999 checkpoint to be finalized and the trainer to exit, then serves it on GPU 0 of job
# 17356154 port 8000 (keep-alive 2668788 untouched).
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; exp=v6_task1B6_20260909_r3; cfg=pi05_yam_mem_v6_task1B6
ck=$root/v6/checkpoints/$cfg/$exp/1999
log=$root/v6/diagnostics/server_v6_b6_1999_hgx1.log
while ! { [ -e "$ck/params" ] && grep -q "\[step=1999\] CheckpointManager Save Finalize is done on all hosts" "$root/v6/logs/train_$exp.log" && tail -1 "$root/v6/logs/train_${exp}_status.log" | grep -q "^exit="; }; do sleep 20; done
echo "$(date '+%m/%d %H:%M') checkpoint 1999 final and trainer exited ($(tail -1 $root/v6/logs/train_${exp}_status.log)); launching server" | tee -a "$log"
cd $root/openpi && JOB=17356154 GRES=1 GPU=0 CPUS=12 NO_PLACEHOLDER=1 SERVE_EXTRA="--num-steps ${NUM_STEPS:-10}" LOG=$log \
  bash cluster_v6/serve_v6_job_v2.sh "$ck" "$cfg" "${PORT:-8000}"
