#!/usr/bin/env bash
# One-off (2026-09-22): upload the dataset from the node-local copy on iris-hgx-1 as a CPU-only srun step of $JOB.
# Usage ON iris-hgx-1:  JOB=17533970 SRC=/scr/kewalk/beans0922/bean_scoop_0905_v5 bash beans/ablations/hf_upload_dataset_node.sh
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export HOME=/iris/u/kewalk MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LOG="$ROOT/beans/ablations/hf_upload_dataset.log"
echo "start $(date) host=$(hostname) job=${JOB:-none} src=${SRC:-default}" >> "$LOG"
if [ -n "${JOB:-}" ]; then
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env CUDA_VISIBLE_DEVICES= \
    "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" dataset ${SRC:+--src "$SRC"} >> "$LOG" 2>&1
else
  CUDA_VISIBLE_DEVICES= "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" dataset ${SRC:+--src "$SRC"} >> "$LOG" 2>&1
fi
echo "exit=$? $(date)" >> "$LOG"
