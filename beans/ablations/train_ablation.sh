#!/usr/bin/env bash
# Generic launcher for ONE beans0922 ablation training on the GPUs you own (4 cards by default). Called by run_<ablation>.sh.
#   CFG=<train config> EXP=<experiment dir name> [MODE=train|smoke] [JOB=<slurm job id>] [GPUS=0,1,2,3] [BATCH=16]
#   [BATCH_FALLBACK="12 8"] [WORKERS=16] [WAIT_FOR=<path>] bash beans/ablations/train_ablation.sh
# - JOB set (this cluster): an `srun --overlap` step inside that allocation pinned to CUDA_VISIBLE_DEVICES=$GPUS.
#   JOB unset (another cluster / a node you already own): python runs directly.
# - Waits until WAIT_FOR exists (default: the KI base checkpoint the memory run warm-starts from) and until nobody holds
#   > 2 GB on the cards (1 GB keep-alives are fine), then trains; on RESOURCE_EXHAUSTED retries with the next batch.
# - Resume: an experiment dir with a numeric checkpoint -> --resume; an empty dir -> --overwrite; none -> fresh.
# - MODE=smoke: <CFG>_smoke, 2 updates, no W&B, no wait for the base beyond its existence check.
set -u
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT/openpi" || exit 2
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92} OPENPI_0920_REMAT=${REMAT:-nothing_saveable}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export HOME=${HOME:-$ROOT}
# the tree keeps every cache under its own root (project_paths.configure_v35_runtime_environment) and refuses inherited
# machine-wide settings, so drop them here (TMPDIR / WANDB_DIR may differ and are kept)
unset HF_HOME HF_LEROBOT_HOME HF_DATASETS_CACHE OPENPI_DATA_HOME OPENPI_JAX_CACHE_DIR UV_CACHE_DIR
CFG=${CFG:?set CFG}; EXP=${EXP:?set EXP}; MODE=${MODE:-train}
GPUS=${GPUS:-0,1,2,3}; NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); WORKERS=${WORKERS:-16}; WANDB=${WANDB:-1}
BATCH=${BATCH:-16}; FALLBACK=${BATCH_FALLBACK:-"12 8 4"}  # 4 x 80 GB (H100): expect 8-12; 4 x 141 GB (H200): 16+
if [ "$MODE" = smoke ]; then CFG="${CFG}_smoke"; EXP="smoke_${EXP}"; WANDB=0; FALLBACK=${BATCH_FALLBACK:-}; fi
BASE_PARAMS="${OPENPI_BEANS_BASE_PARAMS:-$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params}"
WAIT_FOR=${WAIT_FOR:-$BASE_PARAMS}
LOGS="$ROOT/beans/ablations/logs"; mkdir -p "$LOGS"; status="$LOGS/train_${EXP}_status.log"
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
# nvidia-smi ignores CUDA_VISIBLE_DEVICES: on a shared 8-GPU node the direct path must ask about OUR cards only (-i), or a
# second row on the other four cards would wait for the first one forever. Inside a Slurm step the cgroup already limits the view.
gpu_busy() { local q=(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits)
  if [ -n "${JOB:-}" ]; then srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" "${q[@]}" 2>/dev/null | awk '$1+0>2000' | wc -l
  else nvidia-smi -i "$GPUS" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; fi; }
wait_gpu() { local ok=0; for i in $(seq 1 ${WAIT_ROUNDS:-240}); do if [ "$(gpu_busy)" = "0" ]; then ok=$((ok+1)); [ $ok -ge ${FREE_STREAK:-4} ] && return 0; else ok=0; fi; sleep 30; done; return 1; }
wait_path() { local n=0; while [ ! -e "$1" ]; do [ $n -eq 0 ] && log "waiting for $1"; n=$((n+1)); [ $n -gt ${WAIT_PATH_ROUNDS:-2880} ] && return 1; sleep 30; done; sleep 60; return 0; }  # + 60 s: let the writer finish
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"
run_once() {  # $1 = batch
  local ckdir="$ROOT/beans/checkpoints/$CFG/$EXP" extra=() mode=fresh
  if [ -d "$ckdir" ]; then if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi; fi
  echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=${JOB:-none} gpus=$GPUS fsdp=$NGPU config=$CFG exp=$EXP batch=$1 mode=$mode base=$BASE_PARAMS code=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)" >> "$status"
  find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  local cmd=("$PY" scripts/train.py "$CFG" --exp-name "$EXP" --batch-size "$1" --fsdp-devices "$NGPU" --num-workers "$WORKERS"
             $( [ "$WANDB" = 1 ] && echo --wandb-enabled || echo --no-wandb-enabled ) "${extra[@]}")
  local envs=(OPENPI_BEANS_AB_BATCH="$1" OPENPI_BEANS_AB_FSDP="$NGPU" OPENPI_BEANS_BASE_PARAMS="$BASE_PARAMS")
  if [ -n "${JOB:-}" ]; then
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-24}" --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" "${envs[@]}" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  else
    env CUDA_VISIBLE_DEVICES="$GPUS" "${envs[@]}" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  fi
  local rc=$?; echo "exit=$rc $(date +%m/%d\ %H:%M)" >> "$status"; return $rc
}
log "start mode=$MODE host=$(hostname) job=${JOB:-none} gpus=$GPUS batch=$BATCH fallback='$FALLBACK' cfg=$CFG exp=$EXP wait_for=$WAIT_FOR"
wait_path "$WAIT_FOR" || { log "gave up waiting for $WAIT_FOR"; exit 1; }
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
for b in $BATCH $FALLBACK; do
  log "attempt batch=$b"
  if run_once "$b"; then log "end mode=$MODE batch=$b: exit=0"; exit 0; fi
  if tail -400 "$LOGS/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
    log "batch $b ran out of memory"; rm -rf "$ROOT/beans/checkpoints/$CFG/$EXP"; sleep 30; wait_gpu || exit 1; continue
  fi
  log "end mode=$MODE batch=$b: failed (not an OOM), see beans/ablations/logs/train_${EXP}.log"; exit 1
done
log "end mode=$MODE: every batch ran out of memory"; exit 1
