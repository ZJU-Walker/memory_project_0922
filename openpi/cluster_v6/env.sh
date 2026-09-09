#!/usr/bin/env bash

# v6 runtime environment (cluster_v6/README.md §0). Same contract as cluster_v5/env.sh: every mutable cache
# derives from THIS checkout's root (memory_project_v6), so the v6 worktree has its own JAX compilation cache
# (never shared with a concurrent JAX process). Read-only v5 artefacts are reached through the sanctioned
# top-level link `v5` (project_paths.SHARED_DATA_LINKS); v6 run artefacts live under `v6/`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_v6/env.sh instead of executing it" >&2
  exit 2
fi

v6_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HOME=/iris/u/kewalk
export PYTHONDONTWRITEBYTECODE=1
# shellcheck source=../cluster_v35/env.sh
source "${v6_env_dir}/../cluster_v35/env.sh"
# NOTE: HF_LEROBOT_HOME stays at the v3.5 contract value (${MEMORY_PROJECT_ROOT}/data/lerobot; train.py verifies
# it). The v6 datasets are reached through the data config's explicit lerobot_dataset_root (v6/data/lerobot/...);
# only the converter (cluster_v6/task1/convert_task1v6_hgx1.sh) sets HF_LEROBOT_HOME to the v6 dir itself.

mkdir -p \
  "${MEMORY_PROJECT_ROOT}/v6/assets" \
  "${MEMORY_PROJECT_ROOT}/v6/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/v6/diagnostics" \
  "${MEMORY_PROJECT_ROOT}/v6/logs"

unset v6_env_dir
