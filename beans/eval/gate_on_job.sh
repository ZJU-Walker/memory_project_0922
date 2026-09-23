#!/usr/bin/env bash
# Run the two offline gates for one checkpoint inside a Slurm job you already own (no sbatch): waits for the checkpoint, then
# one `srun --overlap` step over the job's cards runs beans/eval/sbatch_offline_eval.sh (battery on card 0, videos on card 1).
#   JOB=17489557 GRES=2 bash beans/eval/gate_on_job.sh pi05_yam_beans0922_v3 beans0922_v3 1000
# Launch it from a shell ON THE NODE (nohup ... &); a plain-ssh shell there sits in the newest job's cgroup, which is why the
# work itself goes through srun into JOB. Uses the node-local dataset mirror when present (its HF cache already exists there).
set -u
CFG=${1:?config}; EXP=${2:?exp}; STEP=${3:?step}; JOB=${JOB:?slurm job id}; GRES=${GRES:-2}
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
ck="$ROOT/beans/checkpoints/$CFG/$EXP/$STEP/params"; log="$ROOT/beans/eval/gate_${EXP}_${STEP}.out"
n=0; until [ -d "$ck" ]; do [ $n -eq 0 ] && echo "$(date +%m/%d\ %H:%M) waiting for $ck" >> "$log"; n=$((n+1)); sleep 60; done
sleep 90  # let the checkpoint writer finish
echo "$(date +%m/%d\ %H:%M) checkpoint present, starting the gates on job $JOB" >> "$log"
env_args=(CFG="$CFG" EXP="$EXP" CKPTS="$STEP" TOOLS="${TOOLS:-battery videos}" BATTERY_GPU="${BATTERY_GPU:-0}" VIDEO_GPU="${VIDEO_GPU:-1}" FLIP_TOKENS="${FLIP_TOKENS:-sentence}")
[ -e "$ROOT/local/bean_scoop_0905_v5/meta/info.json" ] && env_args+=(OPENPI_BEANS_DATASET_ROOT="$ROOT/local/bean_scoop_0905_v5")
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-16}" --gres=gpu:"$GRES" \
  env "${env_args[@]}" bash "$ROOT/beans/eval/sbatch_offline_eval.sh" >> "$log" 2>&1
echo "$(date +%m/%d\ %H:%M) gates for $EXP/$STEP finished (exit=$?)" >> "$log"
