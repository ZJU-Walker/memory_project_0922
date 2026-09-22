#!/usr/bin/env bash
# train_binfill_ctl.sh status|smoke|start|stop -- the single 0920 BinFill runner instance on the 2 x H100 (run ON iris-hgx-1);
# bracket patterns so pgrep never matches itself. BATCH/WORKERS/JOB/REMAT/MEMFRAC pass through to train_binfill_h100.sh.
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT || exit 2
pids() { pgrep -u kewalk -f "^bash robomme/logs/train_binfill_h100[.]sh$"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "binfill trainings: $(pgrep -u kewalk -f "train[.]py pi05_robomme_0920_binfill" | wc -l)"
          tail -3 robomme/logs/train_binfill_status.log 2>/dev/null ;;
  smoke|start) if [ -n "$(pids)" ]; then echo "already running: $(pids | tr '\n' ' ')"; else
            mode=$([ "$1" = smoke ] && echo smoke || echo train)
            (setsid nohup env MODE=$mode JOB=${JOB:-17489557} BATCH=${BATCH:-8} WORKERS=${WORKERS:-16} MEMFRAC=${MEMFRAC:-0.95} REMAT=${REMAT:-dots_saveable} bash robomme/logs/train_binfill_h100.sh > robomme/logs/train_binfill_run$(date +%H%M%S).out 2>&1 < /dev/null &); sleep 4
            echo "started ($mode): $(pids | tr '\n' ' ')"; fi ;;
  stop)   for p in $(pids); do kill $p && echo "killed runner $p"; done
          for p in $(pgrep -u kewalk -f "train[.]py pi05_robomme_0920_binfill"); do kill $p && echo "killed train.py $p"; done ;;
esac
