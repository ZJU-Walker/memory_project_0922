#!/usr/bin/env bash
# Single-stage PickXtimes memory training (user 09-15 15:33: "direct start stage B and use this for all later training,
# no A/B any more, directly do self write"): the boba2B-style memory stage with OWN write timing + label content, fresh
# memory leaves, straight from the plain pi05 checkpoint 7000 (no KI base stage, no oracle stage A). The sentence head
# starts untrained and is learned inside this stage by the sentence CE. Official labels by default.
# Run ON the job's node:
#   WANDB=1 JOB=17422715 GRES=2 VISIBLE=0,1 GPUS=2 BATCH=2 CPUS=16 EXP=pickxtimes_off_selfwrite_r1 \
#     nohup setsid bash cluster_robomme/selfwrite_pickxtimes.sh > robomme/logs/selfwrite_r1.out 2>&1 < /dev/null &
# BATCH=2 on 2xH100 (one 40-step window per card), 4 on 2xH200. Extra train.py args pass through ($@).
set -u
export JOB="${JOB:?job id}" GRES="${GRES:?gpus of the job}" VISIBLE="${VISIBLE:?e.g. 0,1}" GPUS="${GPUS:-2}" ACCUM="${ACCUM:-1}" CPUS="${CPUS:-16}"
CONFIG="${CONFIG:-pi05_robomme_mem_PickXtimes_off_B_plain7000}"; EXP="${EXP:-pickxtimes_off_selfwrite_r1}"
root=/iris/u/kewalk/memory_project_robomme; cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
status="$root/robomme/logs/selfwrite_pickxtimes.log"
echo "selfwrite start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gres=$GRES gpus=$VISIBLE config=$CONFIG exp=$EXP batch=${BATCH:-2} wandb=${WANDB:-0} code=$(git -C $root rev-parse --short HEAD)" | tee -a "$status"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"$GRES" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv < /dev/null | tee -a "$status"
BATCH="${BATCH:-2}" bash cluster_robomme/run_train.sh "$CONFIG" "$EXP" "$@"
echo "selfwrite end $(date +%m/%d\ %H:%M) exp=$EXP" | tee -a "$status"
