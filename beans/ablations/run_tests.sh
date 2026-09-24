#!/usr/bin/env bash
# CPU regressions, A2->B2 smokes, or per-row short A/B probes. Never deletes experiment data.
set -uo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "$ROOT" || exit 2
export PYTHONDONTWRITEBYTECODE=1
MODE=${1:-cpu}; ROWS=${ROWS:-"snap vis8 vis8s state8 vis8s_add state8_add"}
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"
LOGS="$ROOT/beans/ablations/logs"; mkdir -p "$LOGS"; LOG="$LOGS/run_tests.log"
fails=0
cpu_tests() {
  if (cd openpi && JAX_PLATFORMS=cpu PYTHONPATH=scripts "$PY" -B -m pytest -q -p no:cacheprovider --disable-warnings \
      src/openpi/models/pi0_template_slot_test.py src/openpi/models/sensory_prefill_test.py \
      src/openpi/models/pi0_v0922ab_test.py src/openpi/training/beans0922_ablation_test.py scripts/train_v0922ab_test.py \
      scripts/v5_heldout_visual_test.py \
      src/openpi/models/pi0_v0920_query_context_test.py src/openpi/models/pi0_v0920_v4_token_test.py \
      src/openpi/shared/project_paths_test.py) 2>&1 | tee -a "$LOG"; then
    echo 'cpu: PASS'
  else echo 'cpu: FAIL'; fails=$((fails+1)); fi
}
smoke_tests() {
  for row in $ROWS; do
    if bash "beans/ablations/run_${row}.sh" smoke >> "$LOG" 2>&1; then
      echo "smoke $row A2->B2: PASS"
    else echo "smoke $row: FAIL (see train_smoke_slot_${row}_A/B.log)"; fails=$((fails+1)); fi
  done
}
probe_tests() {
  for row in $ROWS; do
    if A_STEPS=${PROBE_A_STEPS:-10} STEPS=${PROBE_STEPS:-100} RUN_NAME="probe_slot_$row" bash "beans/ablations/run_${row}.sh" >> "$LOG" 2>&1; then
      echo "probe $row: PASS"
    else echo "probe $row: FAIL"; fails=$((fails+1)); fi
  done
  "$PY" beans/ablations/probe_report.py --rows $ROWS --no-wandb
}
case "$MODE" in
  cpu) cpu_tests ;; smoke) smoke_tests ;; probe) probe_tests ;;
  all) cpu_tests; smoke_tests; probe_tests ;;
  *) echo "usage: $0 cpu|smoke|probe|all"; exit 2 ;;
esac
echo "done mode=$MODE failures=$fails"; exit "$fails"
