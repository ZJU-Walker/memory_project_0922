#!/bin/bash
# LeRobot conversion of boba_0911 (56 episodes / 259188 frames, 30 Hz, full 640x480, labels v1 = BOBA_LABELS.md).
# Output: memory_project_v6/v6/data/lerobot/yam/boba_0911_v1 (~250 GB at ~4.5 GB/episode; user 2026-09-12 01:59:
# "i still want the data in /iris/u/kewalk", disk cleaned by the user first). Run as an srun --overlap step on a
# compute node; the step writes its own log so a dead srun client is harmless.
set -euo pipefail
LOG=/iris/u/kewalk/memory_project_v6/v6/logs/convert_boba_v1.log
exec >> "$LOG" 2>&1
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME=/iris/u/kewalk/memory_project_v6/v6/data/lerobot
cd /iris/u/kewalk/memory_project_v6/openpi
find examples/yam src/openpi -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] host $(hostname) job ${SLURM_JOB_ID:-none} start boba conversion"
.venv/bin/python examples/yam/convert_yam_data_to_lerobot.py \
  --episode-manifest /iris/u/kewalk/memory_project/data/0911_boba_episode_manifest_v1.json \
  --repo-name yam/boba_0911_v1 "$@"
rc=$?
echo "[$(date)] conversion finished rc=$rc"
touch /iris/u/kewalk/memory_project_v6/v6/logs/convert_boba_v1.done
