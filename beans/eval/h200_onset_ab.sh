#!/bin/bash
# onset A/B on H100 card 0 of job 17425063: python scripts/v5_onset_ab.py for one training episode of v4c/1750.
cd /iris/u/kewalk/memory_project_beans0922/openpi || exit 2
export MEMORY_PROJECT_ROOT=/iris/u/kewalk/memory_project_beans0922 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/iris/u/kewalk
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
R=$MEMORY_PROJECT_ROOT
export HF_HOME="$R/v35/cache/huggingface" HF_DATASETS_CACHE="$R/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$R/data/lerobot"
export OPENPI_DATA_HOME="$R/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$R/v35/cache/jax"
[ -e "$R/local/bean_scoop_0905_v5/meta/info.json" ] && export OPENPI_BEANS_DATASET_ROOT="$R/local/bean_scoop_0905_v5"
EP=${1:-3}; STEP=${STEP:-1750}
srun --jobid=17425063 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:4 env CUDA_VISIBLE_DEVICES=${GPU:-0} \
  .venv/bin/python scripts/v5_onset_ab.py --config-name pi05_yam_beans0922_v4c \
  --params "$R/beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c/$STEP/params" --episode-index "$EP" \
  --manifest "$R/openpi/cluster_v5/beans/beans_episode_manifest_0905_v1.json" \
  --sidecar "$R/openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json" ${EXTRA:-} > "$R/beans/eval/onset_ab_ep$(printf %02d $EP)_${STEP}${TAG:-}.log" 2>&1
echo "onset A/B ep$EP exit=$? $(date +%H:%M)" >> "$R/beans/eval/battery_in_job_status.log"
