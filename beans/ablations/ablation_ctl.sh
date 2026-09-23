#!/usr/bin/env bash
# ablation_ctl.sh status | stop [<row>] -- runner / training processes of the beans0922 ablations (bracket patterns: pgrep
# never matches itself; only ablation processes are touched). `stop vis8` stops that row only (its runner + its train.py),
# `stop` alone stops every row. Start = the run_<row>.sh scripts (detach them yourself:
#   (setsid nohup env GPUS=0,1,2,3 bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 < /dev/null &)
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT" || exit 2
ROW=${2:-}
pids() { pgrep -u "$USER" -f "bash .*beans/ablations/(train_ablation[.]sh|run_${ROW:-[a-z0-9_]+}[.]sh)" | while read -r p; do
  [ -z "$ROW" ] || tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -Eq "EXP=(smoke_)?ab_${ROW} " || tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "run_${ROW}[.]sh" || continue; echo "$p"; done; }
trains() { pgrep -u "$USER" -f "train[.]py pi05_yam_beans0922_ab_${ROW:-[a-z0-9_]+}(_smoke)? "; }  # exact row: vis8 != vis8s
# data-loader workers are SPAWNED processes (cmdline "... -c from multiprocessing.spawn import spawn_main ..."): when the trainer
# is killed from outside (the host-RAM cgroup on Anvil, 09-23) they survive as orphans (parent pid 1), each holding a few GB
# inside the job -- enough to get the next launch killed at startup. `stop` (and `status`) list them; `stop` kills them.
orphans() { pgrep -u "$USER" -f "multiprocessing[.]spawn import spawn_main" | while read -r p; do [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = 1 ] && echo "$p"; done; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "ablation trainings: $(trains | wc -l)"; echo "orphaned loader workers: $(orphans | wc -l) (stop kills them)"; for f in beans/ablations/logs/train_*${ROW}*_status.log; do [ -f "$f" ] && { echo "-- $f"; tail -3 "$f"; }; done ;;
  stop) for p in $(pids); do kill $p && echo "killed runner $p"; done; for p in $(trains); do kill $p && echo "killed train.py $p"; done
        sleep 3; for p in $(orphans); do kill -9 $p && echo "killed orphaned loader worker $p"; done ;;
  *) echo "usage: $0 status|stop [<row>]"; exit 2 ;;
esac
