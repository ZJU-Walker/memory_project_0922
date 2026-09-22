#!/usr/bin/env bash
# beans0922_ctl.sh status|smoke_base|smoke_mem|base|mem|start|stop -- one detached runner for the beans0922 line (start = the chain).
# Bracket patterns so pgrep never matches itself; only beans0922 processes are ever touched.
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT" || exit 2
pids() { pgrep -u "$USER" -f "bash .*beans/logs/(train_beans0922|chain_beans0922)[.]sh"; }
trains() { pgrep -u "$USER" -f "train[.]py pi05_yam_beans0922"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "beans0922 trainings: $(trains | wc -l)"; tail -4 beans/logs/train_beans0922_status.log 2>/dev/null ;;
  smoke_base|smoke_mem|base|mem|start)
     if [ -n "$(pids)" ]; then echo "already running: $(pids | tr '\n' ' ')"; else
       if [ "$1" = start ]; then script=beans/logs/chain_beans0922.sh; mode=chain; else script=beans/logs/train_beans0922.sh; mode=$1; fi
       (setsid nohup env MODE=$mode bash $script > beans/logs/beans0922_run$(date +%m%d_%H%M%S).out 2>&1 < /dev/null &); sleep 3
       echo "started ($mode): $(pids | tr '\n' ' ')"; fi ;;
  stop) for p in $(pids); do kill $p && echo "killed runner $p"; done; for p in $(trains); do kill $p && echo "killed train.py $p"; done ;;
esac
