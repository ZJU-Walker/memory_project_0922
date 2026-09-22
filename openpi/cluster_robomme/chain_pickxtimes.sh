#!/usr/bin/env bash
# PickXtimes chain on two GPUs of a Slurm job: KI base continuation (5k steps, batch 16) then the boba2B-style memory
# stage (2000 updates, batch 4; BASE_CONFIG / B_CONFIG select the label version and blinding). Run ON the job's node, e.g.:
#   JOB=17403682 GRES=4 VISIBLE=2,3 nohup setsid bash cluster_robomme/chain_pickxtimes.sh > robomme/logs/chain_pickxtimes_r1.out 2>&1 &
# Never a pattern kill on this node; the job's 1 GB train_hs.py keep-alives stay.
set -u
export JOB="${JOB:?job id}" GRES="${GRES:?gpus of the job}" VISIBLE="${VISIBLE:?e.g. 2,3}" GPUS="${GPUS:-2}" ACCUM=1 CPUS="${CPUS:-16}"
root=/iris/u/kewalk/memory_project_robomme; cd "$root/openpi" || exit 2
export HOME=/iris/u/kewalk
# 09-15 (user 15:00): the OFFICIAL labels are the plan -> configs with the _off infix (see robomme_config.py); the v2
# two-phase configs (no infix) and the blinded variants (_B_blind) stay selectable through BASE_CONFIG / B_CONFIG.
BASE_CONFIG="${BASE_CONFIG:-pi05_robomme_PickXtimes_base_ki_off}"; BASE_EXP="${BASE_EXP:-pickxtimes_base_ki_off_r1}"
B_CONFIG="${B_CONFIG:-pi05_robomme_mem_PickXtimes_off_B}"; B_EXP="${B_EXP:-pickxtimes_off_memB_r1}"; BASE_STEP="${BASE_STEP:-4999}"
SKIP_BASE="${SKIP_BASE:-0}"
status="$root/robomme/logs/chain_pickxtimes.log"
echo "chain start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gres=$GRES gpus=$VISIBLE base=$BASE_CONFIG/$BASE_EXP B=$B_CONFIG/$B_EXP skip_base=$SKIP_BASE code=$(git -C $root rev-parse --short HEAD)" | tee -a "$status"
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:"$GRES" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$status"
base_ck="$root/robomme/checkpoints/$BASE_CONFIG/$BASE_EXP/$BASE_STEP/params"
if [ "$SKIP_BASE" != "1" ]; then
  BATCH="${BASE_BATCH:-16}" bash cluster_robomme/run_train.sh "$BASE_CONFIG" "$BASE_EXP"
fi
if [ ! -d "$base_ck" ]; then echo "base checkpoint missing ($base_ck) $(date +%m/%d\ %H:%M); B not launched" | tee -a "$status"; exit 3; fi
echo "base done $(date +%m/%d\ %H:%M): $base_ck; launching B $B_EXP" | tee -a "$status"
export OPENPI_ROBOMME_BASE_PARAMS="$base_ck"
BATCH="${B_BATCH:-4}" bash cluster_robomme/run_train.sh "$B_CONFIG" "$B_EXP"
echo "chain end $(date +%m/%d\ %H:%M)" | tee -a "$status"
