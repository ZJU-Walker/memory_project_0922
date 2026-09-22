#!/usr/bin/env bash
# Rebuild the LED bean-scoop LeRobot dataset yam/bean_scoop_0905_v5 (deleted with the v5 checkpoints) exactly as on 2026-09-05
# 23:05 (v5/diagnostics/convert_beans_0905.log): manifest mode over the 89 raw demos, into the v5-private LeRobot root that
# every config pins. CPU only (ffmpeg/opencv), ~25 min. Run ON iris-hgx-1 (the login node killed the 0902 run).
set -u
cd /iris/u/kewalk/memory_project_v5/openpi || exit 2
source cluster_v5/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk HF_LEROBOT_HOME=/iris/u/kewalk/memory_project_v5/v5/data/lerobot CUDA_VISIBLE_DEVICES=
mkdir -p "$HF_LEROBOT_HOME"
echo "convert start $(date '+%m/%d %H:%M') host=$(hostname)"
.venv/bin/python examples/yam/convert_yam_data_to_lerobot.py \
  --episode-manifest /iris/u/kewalk/memory_project/data/0905beans_episode_manifest_v1.json \
  --repo-name yam/bean_scoop_0905_v5 "$@"
echo "convert exit=$? $(date '+%m/%d %H:%M')"
