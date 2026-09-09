#!/bin/bash
# Norm stats for the v6 task1 dataset (yam/task1_find_0908_v6). compute_norm_stats.py writes under
# <assets_base_dir>/<config name>/<repo_id>; the task1 DATA config reads v6/assets/pi05_yam_task1_0908_v6/<repo_id>,
# so the result is copied there (same convention as cluster_v5/launch_beans_hgx1.sh). CPU only; run on a node.
set -euo pipefail
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
cd /iris/u/kewalk/memory_project_v6/openpi
CONFIG=${CONFIG:-pi05_yam_mem_v6_task1A}
REPO=yam/task1_find_0908_v6
echo "[$(date)] host $(hostname) norm stats for $CONFIG"
.venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG" --max-frames 3000
SRC=/iris/u/kewalk/memory_project_v6/v6/assets/$CONFIG/$REPO
DST=/iris/u/kewalk/memory_project_v6/v6/assets/pi05_yam_task1_0908_v6/$REPO
mkdir -p "$(dirname "$DST")"
rm -rf "$DST" && cp -r "$SRC" "$DST"
ls -la "$DST"
echo "[$(date)] done"
touch /iris/u/kewalk/memory_project_v6/v6/logs/norm_stats_task1v6.done
