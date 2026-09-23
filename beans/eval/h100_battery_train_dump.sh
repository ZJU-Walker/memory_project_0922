#!/bin/bash
# after the input-ablation set: the count battery on the TRAIN split (training sampler windows) for v4c/2000 -- per first-go
# step: true count, predicted count (variant CE ranking), count_in_window (blinks inside the window = own-timing bank).
cd /iris/u/kewalk/memory_project_beans0922/openpi || exit 2

R=/iris/u/kewalk/memory_project_beans0922
export MEMORY_PROJECT_ROOT=$R PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
export HF_HOME="$R/v35/cache/huggingface" HF_DATASETS_CACHE="$R/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$R/data/lerobot"
export OPENPI_DATA_HOME="$R/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$R/v35/cache/jax"
[ -e "$R/local/bean_scoop_0905_v5/meta/info.json" ] && export OPENPI_BEANS_DATASET_ROOT="$R/local/bean_scoop_0905_v5"
out="$R/beans/eval/flip_beans0922_v4c_2000_traindump"; rm -rf "$out"
srun --jobid=17489557 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 env CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/v5_count_flip_eval.py --config-name pi05_yam_beans0922_v4c \
  --params "$R/beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c/2000/params" --split train --batches 8 --dump-windows --batch-size 2 \
  --tokens sentence --output-dir "$out" > "$out.log" 2>&1
echo "battery TRAIN dump v4c/2000 exit=$? $(date +%H:%M)" >> "$R/beans/eval/battery_in_job_status.log"
