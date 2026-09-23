#!/bin/bash
# three onset A/B runs in sequence on H100 card 0: 1750 without the pointer bonus; v4/500 (archived) with and without it.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
R=/iris/u/kewalk/memory_project_beans0922
EXTRA="--pointer-beta 0" bash beans/eval/h100_onset_ab.sh 3
mv beans/eval/onset_ab_ep03_1750.log beans/eval/onset_ab_ep03_1750_beta0.log
mkdir -p beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c_at500 && [ -d beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c_at500/500 ] || cp -al beans/checkpoints/archive/v4_500 beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c_at500/500
cd openpi
export MEMORY_PROJECT_ROOT=$R PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
export HF_HOME="$R/v35/cache/huggingface" HF_DATASETS_CACHE="$R/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$R/data/lerobot"
export OPENPI_DATA_HOME="$R/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$R/v35/cache/jax"
[ -e "$R/local/bean_scoop_0905_v5/meta/info.json" ] && export OPENPI_BEANS_DATASET_ROOT="$R/local/bean_scoop_0905_v5"
for beta in default 0; do
  extra=""; [ "$beta" = 0 ] && extra="--pointer-beta 0"
  srun --jobid=17489557 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 env CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python scripts/v5_onset_ab.py --config-name pi05_yam_beans0922_v4c \
    --params "$R/beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c_at500/500/params" --episode-index 3 \
    --manifest "$R/openpi/cluster_v5/beans/beans_episode_manifest_0905_v1.json" \
    --sidecar "$R/openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json" $extra > "$R/beans/eval/onset_ab_ep03_500_beta$beta.log" 2>&1
  echo "onset A/B ep3 ckpt500 beta=$beta exit=$? $(date +%H:%M)" >> "$R/beans/eval/battery_in_job_status.log"
done
