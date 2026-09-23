#!/usr/bin/env bash
# Generic launcher for ONE beans0922 ablation training on the GPUs you own (4 cards by default). Called by run_<ablation>.sh.
#   CFG=<train config> EXP=<experiment dir name> [MODE=train|smoke] [JOB=<slurm job id>] [GPUS=0,1,2,3] [BATCH=16]
#   [BATCH_FALLBACK="12 8"] [STEPS=3000] [WORKERS=16|8 on a <200 GB job] [WAIT_FOR=<path>] bash beans/ablations/train_ablation.sh
# - JOB set (this cluster): an `srun --overlap` step inside that allocation pinned to CUDA_VISIBLE_DEVICES=$GPUS. Two rows
#   on one 4-card job: give both GRES=4 (the step sees all four cards, GPUS pins the pair); a 2-card step would be handed
#   whichever two cards Slurm picks, and GPUS=2,3 would then name cards the step cannot see.
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
# The tree keeps every cache under its own root (project_paths.configure_v35_runtime_environment) and refuses inherited
# machine-wide settings. The `datasets` library fixes its cache dir at IMPORT time, before train.py can set it, so export the
# in-tree spellings here (they equal what train.py would set; a symlinked v35/cache/huggingface/datasets -> local disk is
# followed by both). TMPDIR / WANDB_DIR may differ and are left alone.
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$ROOT/data/lerobot"
export OPENPI_DATA_HOME="$ROOT/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax" UV_CACHE_DIR="$ROOT/v35/cache/uv"
mkdir -p "$HF_HOME" "$OPENPI_DATA_HOME" "$OPENPI_JAX_CACHE_DIR" "$UV_CACHE_DIR" 2>/dev/null; [ -e "$HF_DATASETS_CACHE" ] || mkdir -p "$HF_DATASETS_CACHE"
CFG=${CFG:?set CFG}; EXP=${EXP:?set EXP}; MODE=${MODE:-train}
GPUS=${GPUS:-0,1,2,3}; NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); WANDB=${WANDB:-1}
# Loader workers: 16 unless the Slurm job caps host RAM below 200 GB (Anvil: 4 GB per core, a 32-core job = 128 GB). The
# checkpoint save copies model + optimizer state (~48 GB) to CPU RAM in one go on top of the workers (~3.5 GB each) and the
# runtime (~25 GB): with 16 workers a 128 GB job was SIGKILLed by the cgroup at the first save (09-23). 8 workers still feed
# the cards 2-3x faster than they train.
if [ -z "${WORKERS:-}" ]; then
  WORKERS=16
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    mem=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o -E "mem=[0-9]+[GMT]" | head -1 | sed 's/mem=//')
    case "$mem" in
      *T) mem_gb=$(( ${mem%T} * 1024 )) ;; *G) mem_gb=${mem%G} ;; *M) mem_gb=$(( ${mem%M} / 1024 )) ;; *) mem_gb= ;;
    esac
    if [ -n "$mem_gb" ] && [ "$mem_gb" -lt 200 ]; then WORKERS=8; echo "job memory ${mem}: using 8 loader workers (WORKERS=... to override)"; fi
  fi
fi
STEPS=${STEPS:-${OPENPI_BEANS_AB_STEPS:-3000}}  # updates (the label-write ramp stays 500)
BATCH=${BATCH:-16}; FALLBACK=${BATCH_FALLBACK:-"12 8 4"}  # 4 x 80 GB (H100): expect 8-12; 4 x 141 GB (H200): 16 (32 aborts: per-device attention tensor > 2 GB)
if [ "$MODE" = smoke ]; then CFG="${CFG}_smoke"; EXP="smoke_${EXP}"; WANDB=0; FALLBACK=${BATCH_FALLBACK:-}; fi
# warm start: OPENPI_BEANS_BASE_PARAMS if set, else the LARGEST base step present locally (5000 before the base run has
# finished, 10000 after -- re-run 00_download.sh to fetch the 10000 one), else wait for 10000 to appear (this cluster's chain)
BASE_ROOT="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base"
if [ -z "${OPENPI_BEANS_BASE_PARAMS:-}" ]; then
  latest=$(ls "$BASE_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  BASE_PARAMS="$BASE_ROOT/${latest:-10000}/params"
else BASE_PARAMS="$OPENPI_BEANS_BASE_PARAMS"; fi
WAIT_FOR=${WAIT_FOR:-$BASE_PARAMS}
# node-local dataset mirror (00_download.sh LOCAL_DISK=... or a hand-made link): local/<dataset> is sanctioned by the path guard
if [ -z "${OPENPI_BEANS_DATASET_ROOT:-}" ] && [ -e "$ROOT/local/bean_scoop_0905_v5/meta/info.json" ]; then
  export OPENPI_BEANS_DATASET_ROOT="$ROOT/local/bean_scoop_0905_v5"
fi
LOGS="$ROOT/beans/ablations/logs"; mkdir -p "$LOGS"; status="$LOGS/train_${EXP}_status.log"
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
# nvidia-smi ignores CUDA_VISIBLE_DEVICES: on a shared 8-GPU node the direct path must ask about OUR cards only (-i), or a
# second row on the other four cards would wait for the first one forever. Inside a Slurm step the cgroup already limits the view.
gpu_busy() { local q=(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits)
  if [ -n "${JOB:-}" ]; then srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" "${q[@]}" -i "$GPUS" 2>/dev/null | awk '$1+0>2000' | wc -l
  else nvidia-smi -i "$GPUS" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>2000' | wc -l; fi; }
wait_gpu() { local ok=0; for i in $(seq 1 ${WAIT_ROUNDS:-240}); do if [ "$(gpu_busy)" = "0" ]; then ok=$((ok+1)); [ $ok -ge ${FREE_STREAK:-4} ] && return 0; else ok=0; fi; sleep 30; done; return 1; }
wait_path() { local n=0; while [ ! -e "$1" ]; do [ $n -eq 0 ] && log "waiting for $1"; n=$((n+1)); [ $n -gt ${WAIT_PATH_ROUNDS:-2880} ] && return 1; sleep 30; done; sleep 60; return 0; }  # + 60 s: let the writer finish
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"
discard_if_fresh() {  # a failed attempt leaves a half-written experiment dir; keep it when it holds a checkpoint to resume from
  local d="$ROOT/beans/checkpoints/$CFG/$EXP"
  if ls "$d" 2>/dev/null | grep -qE '^[0-9]+$'; then log "keeping $d (has a checkpoint; the next attempt resumes)"; else rm -rf "$d"; fi
}
run_once() {  # $1 = batch
  local ckdir="$ROOT/beans/checkpoints/$CFG/$EXP" extra=() mode=fresh
  if [ -d "$ckdir" ]; then if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi; fi
  echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=${JOB:-none} gpus=$GPUS fsdp=$NGPU config=$CFG exp=$EXP batch=$1 steps=$STEPS mode=$mode base=$BASE_PARAMS code=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)" >> "$status"
  find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  local cmd=("$PY" scripts/train.py "$CFG" --exp-name "$EXP" --batch-size "$1" --fsdp-devices "$NGPU" --num-workers "$WORKERS"
             $( [ "$WANDB" = 1 ] && echo --wandb-enabled || echo --no-wandb-enabled ) "${extra[@]}")
  local envs=(OPENPI_BEANS_AB_BATCH="$1" OPENPI_BEANS_AB_FSDP="$NGPU" OPENPI_BEANS_AB_STEPS="$STEPS" OPENPI_BEANS_BASE_PARAMS="$BASE_PARAMS")
  if [ -n "${JOB:-}" ]; then
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-24}" --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" "${envs[@]}" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  else
    env CUDA_VISIBLE_DEVICES="$GPUS" "${envs[@]}" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  fi
  local rc=$?; echo "exit=$rc $(date +%m/%d\ %H:%M)" >> "$status"; return $rc
}
log "start mode=$MODE host=$(hostname) job=${JOB:-none} gpus=$GPUS batch=$BATCH fallback='$FALLBACK' cfg=$CFG exp=$EXP base=$BASE_PARAMS dataset=${OPENPI_BEANS_DATASET_ROOT:-default}"
wait_path "$WAIT_FOR" || { log "gave up waiting for $WAIT_FOR"; exit 1; }
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
for b in $BATCH $FALLBACK; do
  log "attempt batch=$b"
  if run_once "$b"; then log "end mode=$MODE batch=$b: exit=0"; exit 0; fi
  if tail -400 "$LOGS/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
    log "batch $b ran out of memory"; discard_if_fresh; sleep 30; wait_gpu || exit 1; continue
  fi
  # an illegal-address abort during compile / autotune at a large batch: a per-device activation crossed the 2 GB kernel
  # indexing limit (seen at batch 32 x 40 ticks x 864 tokens on 4 H200, 09-22); treated like an OOM -> next batch
  if tail -400 "$LOGS/train_${EXP}.log" | grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" && ! grep -q "Step 0:" "$LOGS/train_${EXP}.log"; then
    log "batch $b aborted with CUDA_ERROR_ILLEGAL_ADDRESS before the first step (too large for the kernels)"; discard_if_fresh; sleep 30; wait_gpu || exit 1; continue
  fi
  log "end mode=$MODE batch=$b: failed (not an OOM), see beans/ablations/logs/train_${EXP}.log"; exit 1
done
log "end mode=$MODE: every batch ran out of memory"; exit 1
