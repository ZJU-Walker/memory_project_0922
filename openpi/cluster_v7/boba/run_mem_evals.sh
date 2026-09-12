#!/usr/bin/env bash
# Boba memory dev battery for one checkpoint on ONE GPU of Slurm job $JOB (cluster_v7/README.md §3):
#   JOB=<job> GPU=<idx> [MODES="self oracle"] cluster_v7/boba/run_mem_evals.sh <config-name> <exp-name> <step>
# Rollout videos + summaries of the 3 development episodes (demo11/19/37 = LeRobot indices 9 17 34) with
# scripts/v5_heldout_video.py against the boba manifest/sidecar -> v7/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.*
# Needs a GPU with >= ~20 GB free: the training steps hold 92% of both H200s, so run it on the 2xH100 job once the ctx
# queue is done, or on the H200s between stages.
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:?job id}"; GPU="${GPU:-0}"; modes="${MODES:-self oracle}"
root=/iris/u/kewalk/memory_project_v7
cd "$root/openpi" || exit 2
source cluster_v7/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v7/checkpoints/$config/$exp/$step/params"
manifest="$root/openpi/cluster_v7/boba/boba_episode_manifest_v1.json"
sidecar="$root/openpi/cluster_v7/boba/boba_v5_subtask_labels_v1.json"
out="$root/v7/diagnostics/videos_${exp}_${step}"; mkdir -p "$out"
dev="${DEV:-9 17 34}"
run() { srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:"${GRES:-1}" env CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
[ -e "$ck" ] || { echo "missing params $ck" >> "$out/status.log"; exit 3; }
echo "battery start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gpu=$GPU ck=$ck modes=$modes" >> "$out/status.log"
for mode in $modes; do
  for ep in $dev; do
    tag=$(printf 'ep%02d_%s' "$ep" "$mode")
    [ -e "$out/$tag.json" ] && continue
    run .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M) $(grep -o 'decision steps .*' "$out/${tag}_run.log" | tail -1 | cut -c1-160)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
