#!/usr/bin/env bash
# Test runner for the beans0922 ablations -- run it on any machine that has the repo set up (setup_other_cluster.sh).
#
#   bash beans/ablations/run_tests.sh cpu                 # no GPU: the unit tests of the ablation code (~6 min)
#   GPUS=0,1,2,3 bash beans/ablations/run_tests.sh smoke  # GPU: 2-update smoke of every row (compile + one real batch each)
#   GPUS=0,1,2,3 bash beans/ablations/run_tests.sh probe  # GPU: PROBE_STEPS (100) updates per row, then the stability table
#   bash beans/ablations/run_tests.sh all                 # cpu + smoke + probe
#
# Knobs: ROWS="snap vis8 vis8s vis8s_add state8 state8_add" (default: all six), GPUS (default 0,1,2,3), BATCH (default 16,
# the launcher falls back on OOM), PROBE_STEPS (default 100), WANDB=0 to keep the probes off W&B, JOB=<slurm job id> on a
# Slurm node. Needs the dataset + base checkpoint from 00_download.sh for smoke/probe. Every step appends to
# beans/ablations/logs/run_tests.log; the exit code is non-zero when anything failed.
set -u
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT" || exit 2
export HOME=${HOME:-$ROOT} PYTHONDONTWRITEBYTECODE=1
MODE=${1:-cpu}; ROWS=${ROWS:-"snap vis8 vis8s vis8s_add state8 state8_add"}; PROBE_STEPS=${PROBE_STEPS:-100}
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"; LOGS="$ROOT/beans/ablations/logs"; mkdir -p "$LOGS"; LOG="$LOGS/run_tests.log"
say() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
fails=0

cpu_tests() {
  say "cpu: unit tests (ablation model gates, configs, telemetry keys, question context)"
  ( cd openpi && PYTHONPATH=scripts "$PY" -m pytest -q -p no:cacheprovider \
      src/openpi/models/pi0_v0922ab_test.py src/openpi/training/beans0922_ablation_test.py scripts/train_v0922ab_test.py \
      src/openpi/models/pi0_v0920_query_context_test.py src/openpi/shared/project_paths_test.py 2>&1 | grep -v -E "DeprecationWarning|debug_info|jnp.shape|linear_util|^\s*$" ) | tee -a "$LOG" | tail -n 4
  if grep -q -E "^[0-9]+ passed" <(tail -n 5 "$LOG") && ! tail -n 5 "$LOG" | grep -q -E "[0-9]+ (failed|error)"; then say "cpu: PASS"; else say "cpu: FAIL"; fails=$((fails+1)); fi
}

smoke_tests() {
  for row in $ROWS; do
    say "smoke $row (config pi05_yam_beans0922_ab_${row}_smoke, 2 updates)"
    rm -rf "beans/checkpoints/pi05_yam_beans0922_ab_${row}_smoke/smoke_ab_${row}"
    BATCH_FALLBACK="" bash "beans/ablations/run_${row}.sh" smoke >> "$LOG" 2>&1; rc=$?
    line=$(grep -E "Step 0:" "$LOGS/train_smoke_ab_${row}.log" 2>/dev/null | tail -n 1 | cut -c1-160)
    if [ $rc -eq 0 ] && [ -n "$line" ]; then say "smoke $row: PASS  $line"; else say "smoke $row: FAIL (exit $rc; see $LOGS/train_smoke_ab_${row}.log)"; fails=$((fails+1)); fi
  done
}

probe_tests() {
  for row in $ROWS; do
    say "probe $row ($PROBE_STEPS updates, exp probe_${row})"
    rm -rf "beans/checkpoints/pi05_yam_beans0922_ab_${row}/probe_${row}"
    STEPS=$PROBE_STEPS EXP="probe_${row}" bash "beans/ablations/run_${row}.sh" >> "$LOG" 2>&1; rc=$?
    n=$(grep -c -E "Step [0-9]+:" "$LOGS/train_probe_${row}.log" 2>/dev/null)
    if [ $rc -eq 0 ] && [ "${n:-0}" -ge 2 ]; then say "probe $row: PASS ($n logged steps)"; else say "probe $row: FAIL (exit $rc; see $LOGS/train_probe_${row}.log)"; fails=$((fails+1)); fi
  done
  say "probe: stability table (grad norms, losses, sensory-bank curves; flags at the end)"
  # shellcheck disable=SC2086
  "$PY" beans/ablations/probe_report.py --rows $ROWS $( [ "${WANDB:-1}" = 1 ] || echo --no-wandb ) 2>&1 | tee -a "$LOG"
}

case "$MODE" in
  cpu) cpu_tests ;;
  smoke) smoke_tests ;;
  probe) probe_tests ;;
  all) cpu_tests; smoke_tests; probe_tests ;;
  *) echo "usage: $0 cpu|smoke|probe|all"; exit 2 ;;
esac
say "done mode=$MODE failures=$fails"; exit $fails
