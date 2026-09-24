#!/usr/bin/env bash
# One template-slot stage. Never changes batch on OOM; the A/B chain owns resume.
set -euo pipefail
ROOT=${MEMORY_PROJECT_ROOT:?}; cd "$ROOT/openpi"
CFG=${CFG:?}; EXP=${EXP:?}; GPUS=${GPUS:-0,1,2,3}; MODE=${MODE:-train}
IFS=, read -ra gpu_list <<< "$GPUS"; NGPU=${#gpu_list[@]}
export OPENPI_BEANS_AB_FSDP="$NGPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.92} OPENPI_0920_REMAT=${REMAT:-nothing_saveable}
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets"
export HF_LEROBOT_HOME="$ROOT/data/lerobot" OPENPI_DATA_HOME="$ROOT/v35/cache/openpi"
export OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax" UV_CACHE_DIR="$ROOT/v35/cache/uv"
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
if [ -z "${OPENPI_BEANS_DATASET_ROOT:-}" ] && [ -e "$ROOT/local/bean_scoop_0905_v5/meta/info.json" ]; then
  export OPENPI_BEANS_DATASET_ROOT="$ROOT/local/bean_scoop_0905_v5"
fi
prefix=()
if [ -n "${JOB:-}" ]; then prefix=(srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-24}" --gres=gpu:"${GRES:-$NGPU}"); fi
clear=0
for ((round=0; round<${WAIT_ROUNDS:-120}; round++)); do
  usage=$("${prefix[@]}" nvidia-smi -i "$GPUS" --query-compute-apps=used_memory --format=csv,noheader,nounits)
  if ! awk '$1+0>2000 {busy=1} END {exit !busy}' <<< "$usage"; then clear=1; break; fi
  echo "Waiting for GPUs $GPUS (1 GB keep-alives are allowed)"; sleep 30
done
[ "$clear" = 1 ] || { echo 'GPUs still occupied'; exit 1; }
extra=(); ckpt="${OPENPI_BEANS_AB_CHECKPOINT_ROOT:-$ROOT/beans/checkpoints}/$CFG/$EXP"
if [ -d "$ckpt" ]; then
  if find "$ckpt" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | read -r _; then extra=(--resume); else extra=(--overwrite); fi
fi
wandb=--wandb-enabled; if [ "$MODE" = smoke ] || [ "${WANDB:-1}" = 0 ]; then wandb=--no-wandb-enabled; fi
log="$ROOT/beans/ablations/logs/train_${EXP}.log"
printf 'launch %s config=%s exp=%s GPUs=%s batch=%s code=%s\n' "$(date -Is)" "$CFG" "$EXP" "$GPUS" "$OPENPI_BEANS_AB_BATCH" "$(git -C "$ROOT" rev-parse --short HEAD)" | tee -a "$log"
exec "${prefix[@]}" /usr/bin/env CUDA_VISIBLE_DEVICES="$GPUS" "${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}" scripts/train.py "$CFG" \
  --exp-name "$EXP" --batch-size "$OPENPI_BEANS_AB_BATCH" --fsdp-devices "$NGPU" \
  --gradient-accumulation-steps "$OPENPI_BEANS_AB_ACCUM" \
  --num-workers "$OPENPI_BEANS_AB_WORKERS" "$wandb" "${extra[@]}" >> "$log" 2>&1
