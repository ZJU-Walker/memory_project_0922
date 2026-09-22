#!/usr/bin/env bash
# train_0920_v1_ctl.sh status|smoke|probe|start|stop -- the single 0920_v1 runner (run ON the node of $JOB); bracket patterns
# so pgrep never matches itself; patterns name v1 only (v0 / binfill runners are never touched).
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT || exit 2
pids() { pgrep -u kewalk -f "^bash robomme/logs/train_0920_v1_h200[.]sh$"; }
trains() { pgrep -u kewalk -f "train[.]py pi05_robomme_0920_v1"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "v1 trainings: $(trains | wc -l)"; tail -3 robomme/logs/train_0920_v1_status.log 2>/dev/null ;;
  smoke|probe|start) if [ -n "$(pids)" ]; then echo "already running: $(pids | tr '\n' ' ')"; else
            mode=$([ "$1" = start ] && echo train || echo $1)
            (setsid nohup env MODE=$mode JOB=${JOB:-17422727} BATCH=${BATCH:-32} BATCH_FALLBACK="${BATCH_FALLBACK:-}" WORKERS=${WORKERS:-24} MEMFRAC=${MEMFRAC:-0.95} bash robomme/logs/train_0920_v1_h200.sh > robomme/logs/train_0920_v1_run$(date +%H%M%S).out 2>&1 < /dev/null &); sleep 4
            echo "started ($mode): $(pids | tr '\n' ' ')"; fi ;;
  stop)   for p in $(pids); do kill $p && echo "killed runner $p"; done
          for p in $(trains); do kill $p && echo "killed train.py $p"; done ;;
esac
