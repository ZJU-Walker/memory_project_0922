#!/usr/bin/env bash
# Generic RoboMME launcher (same contract as cluster_v7/train.sh):
#   cluster_robomme/train.sh <config-name> --exp-name <exp> [extra train.py args]
set -euo pipefail
launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${launcher_dir}/env.sh"
if [[ $# -lt 1 ]]; then
  echo "usage: cluster_robomme/train.sh <config-name> --exp-name <exp> [args]" >&2
  exit 2
fi
config="$1"; shift
python_bin="${ROBOMME_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "RoboMME Python is not executable: ${python_bin}" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
exec "${python_bin}" scripts/train.py "${config}" "$@"
