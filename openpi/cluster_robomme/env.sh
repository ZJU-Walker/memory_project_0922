#!/usr/bin/env bash
# RoboMME runtime environment (cluster_robomme/README.md). Same contract as cluster_v7/env.sh: every mutable cache
# derives from THIS checkout's root (memory_project_robomme); the JAX compilation cache is this tree's own. RoboMME
# artefacts live under `robomme/`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_robomme/env.sh instead of executing it" >&2
  exit 2
fi
robomme_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HOME=/iris/u/kewalk
export PYTHONDONTWRITEBYTECODE=1
# shellcheck source=../cluster_v35/env.sh
source "${robomme_env_dir}/../cluster_v35/env.sh"
mkdir -p \
  "${MEMORY_PROJECT_ROOT}/robomme/assets" \
  "${MEMORY_PROJECT_ROOT}/robomme/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/robomme/diagnostics" \
  "${MEMORY_PROJECT_ROOT}/robomme/logs" \
  "${MEMORY_PROJECT_ROOT}/robomme/rollouts"
unset robomme_env_dir
