#!/usr/bin/env bash
# status | stop [row] | archive [row] [--apply]; repository-scoped, preserves keep-alives.
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
exec python3 "$ROOT/beans/ablations/manage_runs.py" "${@:-status}" --root "$ROOT"
