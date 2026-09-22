#!/usr/bin/env bash
# One-off (2026-09-22): push the KI base checkpoint to the Hub as it appears -- first step 5000 (so training elsewhere can
# start), then step 10000 replacing it (user 06:20). CPU-only srun steps of $JOB. Usage ON iris-hgx-1:
#   JOB=17533970 bash beans/ablations/hf_upload_base_node.sh
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export HOME=/iris/u/kewalk MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LOG="$ROOT/beans/ablations/hf_upload_base.log"; BASE="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base"
echo "start $(date) host=$(hostname) job=${JOB:-none} waiting for $BASE/{5000,10000}/params" >> "$LOG"
push() {  # $1 = step
  local params="$BASE/$1/params"
  while [ ! -d "$params" ] || [ -z "$(ls "$params" 2>/dev/null)" ]; do sleep 60; done
  sleep 300  # let orbax finish writing (the trainer keeps going; the dir is complete well before the next checkpoint)
  echo "found step $1 $(date): $(du -sh "$params" | cut -f1)" >> "$LOG"
  if [ -n "${JOB:-}" ]; then
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 env CUDA_VISIBLE_DEVICES= \
      "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" base --step "$1" >> "$LOG" 2>&1
  else
    CUDA_VISIBLE_DEVICES= "$ROOT/openpi/.venv/bin/python" "$ROOT/beans/ablations/hf_upload.py" base --step "$1" >> "$LOG" 2>&1
  fi
  echo "step $1 exit=$? $(date)" >> "$LOG"
}
[ -d "$BASE/10000/params" ] || push 5000
push 10000
echo "done $(date)" >> "$LOG"
