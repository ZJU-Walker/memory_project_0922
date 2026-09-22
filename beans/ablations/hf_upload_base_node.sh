#!/usr/bin/env bash
# One-off (2026-09-22): wait for the KI base checkpoint (beans0922_base/10000/params, written by the 2xH100 base run) and push
# it to the Hub as a CPU-only srun step of $JOB. Usage ON iris-hgx-1:  JOB=17533970 bash beans/ablations/hf_upload_base_node.sh
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export HOME=/iris/u/kewalk MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LOG="$ROOT/beans/ablations/hf_upload_base.log"; PARAMS="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params"
echo "start $(date) host=$(hostname) job=${JOB:-none} waiting for $PARAMS" >> "$LOG"
while [ ! -d "$PARAMS" ] || [ ! -f "$PARAMS/_METADATA" ] && [ ! -f "$PARAMS/checkpoint" ] && [ -z "$(ls "$PARAMS" 2>/dev/null)" ]; do sleep 60; done
sleep 300  # let orbax finish writing (the trainer keeps going; the params dir is complete once the next step logs)
echo "found $(date): $(du -sh "$PARAMS" | cut -f1)" >> "$LOG"
if [ -n "${JOB:-}" ]; then
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 env CUDA_VISIBLE_DEVICES= \
    "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" base >> "$LOG" 2>&1
else
  CUDA_VISIBLE_DEVICES= "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" base >> "$LOG" 2>&1
fi
echo "exit=$? $(date)" >> "$LOG"
