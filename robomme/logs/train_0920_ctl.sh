#!/usr/bin/env bash
# train_0920_ctl.sh status|smoke|start|stop -- the single 0920_v0 runner instance (run ON the node of $JOB); bracket
# patterns so pgrep never matches itself.
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT || exit 2
pids() { pgrep -u kewalk -f "^bash robomme/logs/train_0920_v0_h200(_v2)?[.]sh$"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "0920 trainings: $(pgrep -u kewalk -f "train[.]py pi05_robomme_0920" | wc -l)"
          tail -3 robomme/logs/train_0920_status.log 2>/dev/null ;;
  smoke|start) if [ -n "$(pids)" ]; then echo "already running: $(pids | tr '\n' ' ')"; else
            mode=$([ "$1" = smoke ] && echo smoke || echo train)
            (setsid nohup env MODE=$mode JOB=${JOB:-17422727} BATCH=${BATCH:-32} WORKERS=${WORKERS:-24} MEMFRAC=${MEMFRAC:-0.95} bash robomme/logs/train_0920_v0_h200_v2.sh > robomme/logs/train_0920_run$(date +%H%M%S).out 2>&1 < /dev/null &); sleep 4
            echo "started ($mode): $(pids | tr '\n' ' ')"; fi ;;
  stop)   for p in $(pids); do kill $p && echo "killed runner $p"; done
          for p in $(pgrep -u kewalk -f "train[.]py pi05_robomme_0920"); do kill $p && echo "killed train.py $p"; done ;;
esac
