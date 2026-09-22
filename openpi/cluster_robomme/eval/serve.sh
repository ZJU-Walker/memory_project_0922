#!/usr/bin/env bash
# Model side of a RoboMME rollout (this tree's venv, one GPU): cluster_robomme/eval/serve.sh --task PickXtimes --stage B --checkpoint <dir>/<step>
set -euo pipefail
eval_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${eval_script_dir}/../env.sh"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_JAX_CACHE_DIR="${OPENPI_JAX_CACHE_DIR:-${MEMORY_PROJECT_ROOT}/robomme/cache/eval_jax}"   # 09-17: per-process override (two servers on one job must not share a cache)
mkdir -p "${OPENPI_JAX_CACHE_DIR}"
export PYTHONPATH="${MEMORY_PROJECT_ROOT}/openpi/scripts:${eval_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
eval_model_python="${ROBOMME_MODEL_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
exec "${eval_model_python}" -u "${eval_script_dir}/serve_policy.py" "$@"
