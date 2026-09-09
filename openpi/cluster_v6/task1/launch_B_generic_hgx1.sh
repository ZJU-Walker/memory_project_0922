#!/usr/bin/env bash
# Run ON iris-hgx-1: stop the stage-A run, protect its checkpoint <step> as keep_<step>, launch the stage-B config from it.
#   cluster_v6/task1/launch_B_generic_hgx1.sh <A cfg> <A exp> <step> <B cfg> <B exp> <ENV VAR the B config reads>
export HOME=/iris/u/kewalk
acfg="$1"; aexp="$2"; step="$3"; bcfg="$4"; bexp="$5"; envvar="$6"
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; ckroot=$root/v6/checkpoints
log=$logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-1] $*" >> "$log"; }
pat="scripts/train.py pi05_yam_mem_v6_task[1]|run_train_hgx1.sh pi05_yam_mem_v6_task[1]"
say "stopping $aexp for $bexp from ckpt $step"
pids=$(pgrep -u kewalk -f "$pat"); [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 20
left=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task[1]"); [ -n "$left" ] && { kill -KILL $left; sleep 10; }
until ! pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task[1]" >/dev/null; do sleep 5; done
sleep 30
say "stopped; keep-alive 3743806 $([ -d /proc/3743806 ] && echo alive || echo GONE)"
for f in $logs/train_v6_task1*_status.log; do  # the killed wrapper cannot log its own exit: write it for the gate's `exited` check
  [ "$(tail -1 "$f" | cut -c1-6)" = "launch" ] && echo "exit=143 $(date +%H:%M) stopped by $(basename "$0")" >> "$f"
done
[ -d "$ckroot/$acfg/$aexp/keep_$step" ] || cp -r "$ckroot/$acfg/$aexp/$step" "$ckroot/$acfg/$aexp/keep_$step"
[ -e "$ckroot/$acfg/$aexp/keep_$step/params" ] && say "ckpt $step protected as keep_$step" || { say "keep_$step COPY FAILED (stop)"; exit 1; }
export "$envvar=v6/checkpoints/$acfg/$aexp/keep_$step/params"
cd $root/openpi && find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
say "launching $bexp from ${!envvar}"
JOB=17315830 GPUS=4 BATCH=8 setsid nohup bash $cv6/run_train_hgx1.sh "$bcfg" "$bexp" > $logs/launch_${bexp}.out 2>&1 < /dev/null & disown
sleep 5; tail -1 $logs/train_${bexp}_status.log >> "$log"
