#!/usr/bin/env bash
# Offline held-out probe of a beans0922 memory checkpoint: walks each development episode at the training tick (5 frames)
# with the sentence bank carried across ticks, decodes the subtask sentence every tick, writes it to the bank (self mode:
# own sentences, every tick, as in training after the label ramp; oracle mode: the label sentences) and renders the top
# camera video with the ground-truth phase, the decoded sentence and the bank overlaid (scripts/v5_heldout_video.py).
#   [JOB=<slurm job> GPU=<idx> GRES=<cards of that job>] [MODES="self oracle"] [EPISODES="25 29 59 64 72 73"] \
#   bash beans/eval/run_heldout_videos.sh <config-name> <exp-name> <step>
# Outputs: beans/eval/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json,_run.log} + status.log. JOB unset = run python directly.
set -u
config="${1:?config name}"; exp="${2:?exp name}"; step="${3:?checkpoint step}"
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT/openpi" || exit 2
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.6} HOME=${HOME:-$ROOT}
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$ROOT/data/lerobot"
export OPENPI_DATA_HOME="$ROOT/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax"
if [ -z "${OPENPI_BEANS_DATASET_ROOT:-}" ] && [ -e "$ROOT/local/bean_scoop_0905_v5/meta/info.json" ]; then
  export OPENPI_BEANS_DATASET_ROOT="$ROOT/local/bean_scoop_0905_v5"   # node-local mirror (path-guard sanctioned)
fi
ck="$ROOT/beans/checkpoints/$config/$exp/$step/params"; [ -d "$ck" ] || { echo "no checkpoint at $ck"; exit 1; }
manifest="$ROOT/openpi/cluster_v5/beans/beans_episode_manifest_0905_v1.json"
sidecar="$ROOT/openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json"
out="$ROOT/beans/eval/videos_${exp}_${step}"; mkdir -p "$out"
modes="${MODES:-self oracle}"; episodes="${EPISODES:-25 29 59 64 72 73}"   # manifest split "development" (x=2,1,3,3,2,1)
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"
run() { if [ -n "${JOB:-}" ]; then srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${CPUS:-8}" --gres=gpu:"${GRES:-1}" \
          env CUDA_VISIBLE_DEVICES="${GPU:-0}" "$@"; else env CUDA_VISIBLE_DEVICES="${GPU:-0}" "$@"; fi; }
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "start $(date +%m/%d\ %H:%M) host=$(hostname) job=${JOB:-none} gpu=${GPU:-0} ck=$ck modes='$modes' episodes='$episodes' dataset=${OPENPI_BEANS_DATASET_ROOT:-default}" >> "$out/status.log"
for mode in $modes; do
  for ep in $episodes; do
    tag=$(printf 'ep%02d_%s%s' "$ep" "$mode" "${TAGSUF:-}")
    [ -e "$out/$tag.json" ] && [ -e "$out/$tag.mp4" ] && continue
    run "$PY" scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" ${EXTRA:-} > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M) $(grep -o 'decision steps .*' "$out/${tag}_run.log" | tail -1 | cut -c1-160)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
