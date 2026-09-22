#!/usr/bin/env bash
# ablation_ctl.sh status|stop [<exp>] -- runner / training processes of the beans0922 ablations (bracket patterns: pgrep
# never matches itself; only ablation processes are touched). Start = the run_<ablation>.sh scripts (detach them yourself:
#   (setsid nohup env JOB=<id> bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 < /dev/null &)
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT" || exit 2
pids() { pgrep -u "$USER" -f "bash .*beans/ablations/(train_ablation|run_[a-z0-9_]+)[.]sh"; }
trains() { pgrep -u "$USER" -f "train[.]py pi05_yam_beans0922_ab_"; }
case "${1:-status}" in
  status) echo "runner pids: $(pids | tr '\n' ' ')"; echo "ablation trainings: $(trains | wc -l)"; for f in beans/ablations/logs/train_*_status.log; do [ -f "$f" ] && { echo "-- $f"; tail -3 "$f"; }; done ;;
  stop) for p in $(pids); do kill $p && echo "killed runner $p"; done; for p in $(trains); do kill $p && echo "killed train.py $p"; done ;;
esac
