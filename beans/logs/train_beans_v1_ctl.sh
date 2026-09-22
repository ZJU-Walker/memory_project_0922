#!/usr/bin/env bash
# train_beans_v1_ctl.sh status|smoke|start|stop -- the single beans_0920_v1 runner (run ON iris-hgx-1); bracket patterns so pgrep
# never matches itself; the patterns name this run only (v0/v1 RoboMME and BinFill runners are never touched).
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT || exit 2
pids() { pgrep -u kewalk -f "^bash beans/logs/train_beans_v1_h100[.]sh$"; }
trains() { pgrep -u kewalk -f "train[.]py pi05_yam_beans_0920_v1"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "beans trainings: $(trains | wc -l)"; tail -3 beans/logs/train_beans_v1_status.log 2>/dev/null ;;
  smoke|start) if [ -n "$(pids)" ]; then echo "already running: $(pids | tr '\n' ' ')"; else
            mode=$([ "$1" = start ] && echo train || echo smoke)
            (setsid nohup env MODE=$mode JOB=${JOB:-17489557} BATCH=${BATCH:-4} BATCH_FALLBACK="${BATCH_FALLBACK:-2}" WORKERS=${WORKERS:-12} MEMFRAC=${MEMFRAC:-0.92} bash beans/logs/train_beans_v1_h100.sh > beans/logs/train_beans_v1_run$(date +%H%M%S).out 2>&1 < /dev/null &); sleep 4
            echo "started ($mode): $(pids | tr '\n' ' ')"; fi ;;
  stop)   for p in $(pids); do kill $p && echo "killed runner $p"; done
          for p in $(trains); do kill $p && echo "killed train.py $p"; done ;;
esac
