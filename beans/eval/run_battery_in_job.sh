#!/bin/bash
# run_battery_in_job.sh CFG EXP STEP : the count-flip battery (scripts/v5_count_flip_eval.py, development split, 96 windows)
# on one card of an EXISTING Slurm job, as an --overlap step: JOB=<jobid> GPU=<card index> GRES=<cards of that job> [CPUS=8]
# [FLIP_TOKENS=sentence] [FLIP_BATCH=2] [MEMFRAC=0.9]. Same environment and output layout as sbatch_offline_eval.sh
# (beans/eval/flip_<exp>_<step>_<tokens>/count_flip_eval.json + .log). Launch it FROM the node (setsid nohup), never from sc.
set -u
CFG=${1:?config}; EXP=${2:?exp}; STEP=${3:?step}; JOB=${JOB:?slurm job id}; GPU=${GPU:-0}; GRES=${GRES:-1}
ROOT=${MEMORY_PROJECT_ROOT:-/iris/u/kewalk/memory_project_beans0922}; cd "$ROOT/openpi" || exit 2
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/iris/u/kewalk
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.9}
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$ROOT/data/lerobot"
export OPENPI_DATA_HOME="$ROOT/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax"
target=$(readlink -f "$HF_DATASETS_CACHE" 2>/dev/null || readlink "$HF_DATASETS_CACHE"); mkdir -p "$target" || exit 3
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
FLIP_TOKENS=${FLIP_TOKENS:-sentence}; EVAL="$ROOT/beans/eval"
out="$EVAL/flip_${EXP}_${STEP}_${FLIP_TOKENS}"; log="$EVAL/flip_${EXP}_${STEP}_${FLIP_TOKENS}.log"
[ -e "$out/count_flip_eval.json" ] && { echo "battery $STEP: exists"; exit 0; }
echo "battery $CFG/$EXP/$STEP on host=$(hostname) job=$JOB gpu=$GPU cache=$target $(date +%m/%d\ %H:%M)" | tee -a "$EVAL/battery_in_job_status.log"
rc=1
for bs in ${FLIP_BATCH:-2} 1; do
  rm -rf "$out"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-8}" --gres=gpu:"$GRES" \
    env CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/v5_count_flip_eval.py --config-name "$CFG" \
    --params "$ROOT/beans/checkpoints/$CFG/$EXP/$STEP/params" --split development --batches $((96 / bs)) --batch-size $bs \
    --tokens "$FLIP_TOKENS" --output-dir "$out" > "$log" 2>&1; rc=$?
  [ $rc -eq 0 ] && break
  grep -q RESOURCE_EXHAUSTED "$log" || break
  echo "battery $STEP: batch $bs ran out of memory, retrying smaller" | tee -a "$EVAL/battery_in_job_status.log"
done
echo "battery $CFG/$EXP/$STEP exit=$rc $(date +%m/%d\ %H:%M)" | tee -a "$EVAL/battery_in_job_status.log"
[ -e "$out/count_flip_eval.json" ] && python3 -c "
import json; r=json.load(open('$out/count_flip_eval.json'))
print('battery $STEP first go step:', r['summary_first_go_step'])" | tee -a "$EVAL/battery_in_job_status.log"
exit $rc
