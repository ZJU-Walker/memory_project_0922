#!/usr/bin/env bash
# Generic v6 launcher (same contract as cluster_v5/train.sh):
#   cluster_v6/train.sh <config-name> --exp-name <exp> [extra train.py args]
set -euo pipefail
v6_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v6_launcher_dir}/env.sh"
if [[ $# -lt 1 ]]; then
  echo "usage: cluster_v6/train.sh <config-name> --exp-name <exp> [args]" >&2
  exit 2
fi
v6_config="$1"; shift
v6_python="${V6_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v6_python}" ]]; then
  echo "v6 Python is not executable: ${v6_python}" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
exec "${v6_python}" scripts/train.py "${v6_config}" "$@"
