#!/usr/bin/env bash
# task1 development battery for one checkpoint on ONE GPU of Slurm job $JOB (default 17315830, iris-hgx-1; GPU 0).
#   [JOB=17315830] [GPU=0] [MODES="self oracle"] cluster_v6/task1/run_task1_evals_hgx1.sh <config-name> <exp-name> <step>
# Rollout videos + summaries of the 6 development episodes (demo10 x {spoon,tape}, demo19 x {banana,box},
# demo54 x {banana,spoon}; LeRobot indices 12 13 26 27 67 68) with scripts/v5_heldout_video.py against the v6
# manifest/sidecar -> v6/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json,_run.log}, status.log.
# A decision step counts as correct when the decoded sentence equals the label exactly ("open bin k").
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:-17315830}"; GPU="${GPU:-0}"; modes="${MODES:-self oracle}"
root=/iris/u/kewalk/memory_project_v6
cd "$root/openpi" || exit 2
source cluster_v6/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
ck="$root/v6/checkpoints/$config/$exp/$step/params"
manifest="$root/openpi/cluster_v6/task1/task1v6_episode_manifest_v1.json"
sidecar="$root/openpi/cluster_v6/task1/task1v6_v5_subtask_labels_v1.json"
out="$root/v6/diagnostics/videos_${exp}_${step}"; mkdir -p "$out"
dev="12 13 26 27 67 68"
run() { srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
[ -e "$ck" ] || { echo "missing params $ck" >> "$out/status.log"; exit 3; }
echo "battery start $(date +%m/%d\ %H:%M) host=$(hostname) ck=$ck modes=$modes" >> "$out/status.log"
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
