#!/usr/bin/env bash
# Generic v7 launcher (same contract as cluster_v5/train.sh):
#   cluster_v7/train.sh <config-name> --exp-name <exp> [extra train.py args]
set -euo pipefail
v7_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v7_launcher_dir}/env.sh"
if [[ $# -lt 1 ]]; then
  echo "usage: cluster_v7/train.sh <config-name> --exp-name <exp> [args]" >&2
  exit 2
fi
v7_config="$1"; shift
v7_python="${V7_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v7_python}" ]]; then
  echo "v7 Python is not executable: ${v7_python}" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
exec "${v7_python}" scripts/train.py "${v7_config}" "$@"
