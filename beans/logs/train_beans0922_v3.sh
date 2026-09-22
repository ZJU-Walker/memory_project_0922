#!/usr/bin/env bash
# Portable launcher for the beans0922 line (LED bean scoop, real YAM station). Runs ONE training from the tree this script lives in.
#   v3 copy of train_beans0922.sh (the v1 instance of that file is still running: never edit a running bash script).
#   MODE=mem|smoke_mem [CFG=pi05_yam_beans0922_v3] [EXP=beans0922_v3] [JOB=<slurm job>] [GPUS=0,1] [GRES=<cards of the job>] [BATCH=8] [BATCH_FALLBACK="4"] bash beans/logs/train_beans0922_v3.sh
# - With JOB set (this cluster): an `srun --overlap` step inside that job pinned to CUDA_VISIBLE_DEVICES=$GPUS (--gres must name ALL
#   the job's GPUs, GRES defaults to their count). Without JOB (another cluster / a node you already own): runs python directly.
# - Refuses to start while another process holds > 2 GB on the visible cards (keep-alives of ~1 GB are fine).
# - mem mode: on RESOURCE_EXHAUSTED retries once per batch in BATCH_FALLBACK with a fresh experiment dir.
# - Resume: an experiment dir with a numeric checkpoint -> --resume; an empty dir -> --overwrite; none -> fresh.
set -u
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT/openpi" || exit 2
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92} OPENPI_0920_REMAT=${REMAT:-nothing_saveable}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}  # the wandb service imports from the same (possibly cold, NFS) venv; 30 s is too short
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
MODE=${MODE:-base}; GPUS=${GPUS:-0,1}; NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); WORKERS=${WORKERS:-12}; WANDB=${WANDB:-1}
case $MODE in
  base)       CFG=pi05_yam_beans0922_base;       EXP=${EXP:-beans0922_base}; BATCH=${BATCH:-16}; FALLBACK=${BATCH_FALLBACK:-8} ;;
  smoke_base) CFG=pi05_yam_beans0922_base_smoke; EXP=${EXP:-smoke_base};     BATCH=${BATCH:-16}; FALLBACK=${BATCH_FALLBACK:-}; WANDB=0 ;;
  mem)        CFG=${CFG:-pi05_yam_beans0922_v3};  EXP=${EXP:-beans0922_v3};   BATCH=${BATCH:-8};  FALLBACK=${BATCH_FALLBACK:-4} ;;  # 2 x H200: 8 fits (peer ablation); H100 pair: 4
  smoke_mem)  CFG=${CFG:-pi05_yam_beans0922_v3}_smoke; EXP=${EXP:-smoke_mem_v3}; BATCH=${BATCH:-4}; FALLBACK=${BATCH_FALLBACK:-}; WANDB=0 ;;
  *) echo "MODE must be base|mem|smoke_base|smoke_mem"; exit 2 ;;
esac
LOGS="$ROOT/beans/logs"; mkdir -p "$LOGS"; status="$LOGS/train_beans0922_status.log"
log() { echo "[$(date +%m/%d\ %H:%M:%S)] $*" | tee -a "$status"; }
# look at the job's own cards: a plain shell on a shared node sits in the newest job's cgroup and sees the wrong GPUs
gpu_busy() { local q=(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits)
  if [ -n "${JOB:-}" ]; then srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" "${q[@]}" 2>/dev/null | awk '$1+0>2000' | wc -l
  else CUDA_VISIBLE_DEVICES=$GPUS "${q[@]}" 2>/dev/null | awk '$1+0>2000' | wc -l; fi; }
# free = nobody holds > 2 GB for FREE_STREAK consecutive 30 s polls (default 6 = 3 min), so a pause between someone's eval episodes
# does not count as free; WAIT_ROUNDS polls at most (default 240 = 2 h)
wait_gpu() { local ok=0; for i in $(seq 1 ${WAIT_ROUNDS:-240}); do if [ "$(gpu_busy)" = "0" ]; then ok=$((ok+1)); [ $ok -ge ${FREE_STREAK:-6} ] && return 0; else ok=0; fi; sleep 30; done; return 1; }
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"
run_once() {  # $1 = batch
  local ckdir="$ROOT/beans/checkpoints/$CFG/$EXP" extra=() mode=fresh
  if [ -d "$ckdir" ]; then if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi; fi
  echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=${JOB:-none} gpus=$GPUS fsdp=$NGPU config=$CFG exp=$EXP batch=$1 mode=$mode code=$(git -C "$ROOT" rev-parse --short HEAD)" >> "$LOGS/train_${EXP}_status.log"
  find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  local cmd=("$PY" scripts/train.py "$CFG" --exp-name "$EXP" --batch-size "$1" --fsdp-devices "$NGPU" --num-workers "$WORKERS"
             $( [ "$WANDB" = 1 ] && echo --wandb-enabled || echo --no-wandb-enabled ) "${extra[@]}")
  if [ -n "${JOB:-}" ]; then
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-16}" --gres=gpu:"${GRES:-$NGPU}" env CUDA_VISIBLE_DEVICES="$GPUS" OPENPI_BEANS_BATCH="$1" OPENPI_BEANS_BASE_BATCH="$1" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$GPUS" OPENPI_BEANS_BATCH="$1" OPENPI_BEANS_BASE_BATCH="$1" "${cmd[@]}" >> "$LOGS/train_${EXP}.log" 2>&1
  fi
  local rc=$?; echo "exit=$rc $(date +%m/%d\ %H:%M)" >> "$LOGS/train_${EXP}_status.log"; return $rc
}
log "start mode=$MODE host=$(hostname) job=${JOB:-none} gpus=$GPUS batch=$BATCH fallback='$FALLBACK' cfg=$CFG exp=$EXP memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION"
wait_gpu || { log "cards busy (another process holds > 2 GB)"; exit 1; }
for b in $BATCH $FALLBACK; do
  log "attempt batch=$b"
  if run_once "$b"; then log "end mode=$MODE batch=$b: exit=0"; exit 0; fi
  if tail -400 "$LOGS/train_${EXP}.log" | grep -q "RESOURCE_EXHAUSTED"; then
    log "batch $b ran out of memory"; rm -rf "$ROOT/beans/checkpoints/$CFG/$EXP"; sleep 30; wait_gpu || exit 1; continue
  fi
  log "end mode=$MODE batch=$b: failed (not an OOM), see beans/logs/train_${EXP}.log"; exit 1
done
log "end mode=$MODE: every batch ran out of memory"; exit 1
