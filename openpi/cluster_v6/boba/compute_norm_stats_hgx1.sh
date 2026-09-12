#!/bin/bash
# Norm stats for the boba dataset (yam/boba_0911_v1). compute_norm_stats.py writes under
# <assets_base_dir>/<config name>/<repo_id>; the boba DATA config reads v6/assets/pi05_yam_boba_0911_v1/<repo_id>,
# so the result is copied there (same convention as cluster_v6/task1/compute_norm_stats_hgx1.sh). CPU only; run on
# a node as an srun --overlap step; the step writes its own log.
set -euo pipefail
LOG=/iris/u/kewalk/memory_project_v6/v6/logs/norm_stats_boba_v1.log
exec >> "$LOG" 2>&1
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
cd /iris/u/kewalk/memory_project_v6/openpi
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
CONFIG=${CONFIG:-pi05_yam_boba0911_base}
REPO=yam/boba_0911_v1
echo "[$(date)] host $(hostname) norm stats for $CONFIG"
.venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG" --max-frames 3000
SRC=/iris/u/kewalk/memory_project_v6/v6/assets/$CONFIG/$REPO
DST=/iris/u/kewalk/memory_project_v6/v6/assets/pi05_yam_boba_0911_v1/$REPO
mkdir -p "$(dirname "$DST")"
rm -rf "$DST" && cp -r "$SRC" "$DST"
ls -la "$DST"
echo "[$(date)] done"
touch /iris/u/kewalk/memory_project_v6/v6/logs/norm_stats_boba_v1.done
