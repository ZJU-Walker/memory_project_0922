#!/bin/bash
# LeRobot conversion of the task1 collection with the v6 label convention (restated closing sentence, 16 sentences).
# Same 71 episodes / order / split as yam/task1_find_0908_v5; only the per-frame task strings differ.
set -euo pipefail
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME=/iris/u/kewalk/memory_project_v6/v6/data/lerobot
cd /iris/u/kewalk/memory_project_v6/openpi
find examples/yam src/openpi -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] host $(hostname) start conversion (v6 labels)"
.venv/bin/python examples/yam/convert_yam_data_to_lerobot.py \
  --episode-manifest /iris/u/kewalk/memory_project/data/0908_task1_episode_manifest_v6.json \
  --repo-name yam/task1_find_0908_v6 "$@"
echo "[$(date)] conversion finished rc=$?"
touch /iris/u/kewalk/memory_project_v6/v6/logs/convert_task1v6.done
