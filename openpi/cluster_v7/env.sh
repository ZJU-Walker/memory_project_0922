#!/usr/bin/env bash

# v7 runtime environment (cluster_v7/README.md §0). Same contract as cluster_v5/env.sh: every mutable cache
# derives from THIS checkout's root (memory_project_v7), so the v7 worktree has its own JAX compilation cache
# (never shared with a concurrent JAX process). Read-only v5 artefacts are reached through the sanctioned
# top-level link `v5` (project_paths.SHARED_DATA_LINKS); v7 run artefacts live under `v7/`; v6 artefacts (datasets, the boba base checkpoint) are read through the top-level link `v6`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_v7/env.sh instead of executing it" >&2
  exit 2
fi

v7_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HOME=/iris/u/kewalk
export PYTHONDONTWRITEBYTECODE=1
# shellcheck source=../cluster_v35/env.sh
source "${v7_env_dir}/../cluster_v35/env.sh"
# NOTE: HF_LEROBOT_HOME stays at the v3.5 contract value (${MEMORY_PROJECT_ROOT}/data/lerobot; train.py verifies
# it). The v6 and v7 datasets are reached through the data config's explicit lerobot_dataset_root (v6/data/lerobot/...);
# only the converter (cluster_v6/boba/convert_boba_hgx1.sh) sets HF_LEROBOT_HOME to the v6 dir itself.

mkdir -p \
  "${MEMORY_PROJECT_ROOT}/v7/assets" \
  "${MEMORY_PROJECT_ROOT}/v7/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/v7/diagnostics" \
  "${MEMORY_PROJECT_ROOT}/v7/logs"

unset v7_env_dir
