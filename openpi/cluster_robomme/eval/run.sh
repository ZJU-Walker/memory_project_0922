#!/usr/bin/env bash
# Simulator side of a RoboMME rollout (benchmark venv): cluster_robomme/eval/run.sh --task PickXtimes --split val --episodes 0
set -euo pipefail
eval_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
eval_benchmark="${ROBOMME_BENCHMARK_ROOT:-/iris/u/kewalk/robomme_benchmark}"
eval_sim_python="${ROBOMME_SIM_PYTHON:-${eval_benchmark}/.venv/bin/python}"
export PYTHONPATH="${eval_benchmark}/src:${eval_script_dir}"
exec "${eval_sim_python}" -u "${eval_script_dir}/rollout.py" "$@"
