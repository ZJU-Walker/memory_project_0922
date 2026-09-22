#!/usr/bin/env bash
# RoboMME memory dev battery for one checkpoint on ONE GPU of Slurm job $JOB (cluster_robomme/README.md §4):
#   JOB=<job> GPU=<idx> [MODES="self oracle"] [EPS="0 49 99"] [SIDECAR=<file> TAGSUF=_shift1] cluster_robomme/run_mem_evals.sh <config-name> <exp-name> <step>
# Rollout videos + per-step traces of training episodes (there is no held-out demo split; the benchmark val/test
# rollouts are the real evaluation) with scripts/v5_heldout_video.py against the RoboMME manifest / v2 sidecar
# -> robomme/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.*   Needs ~20 GB on the card (shares it: PREALLOCATE=false).
set -u
config="$1"; exp="$2"; step="$3"; JOB="${JOB:?job id of the eval GPU}"; GPU="${GPU:-0}"; modes="${MODES:-self oracle}"
root=/iris/u/kewalk/memory_project_robomme
cd "$root/openpi" || exit 2
source cluster_robomme/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk XLA_PYTHON_CLIENT_PREALLOCATE=false
# evals never share the training run's JAX compilation cache (NFS cache corrupts under concurrent writers)
export OPENPI_JAX_CACHE_DIR="$root/v35/cache/jax_eval_$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -cs "A-Za-z0-9" "_" | sed "s/_$//")"; mkdir -p "$OPENPI_JAX_CACHE_DIR"
ck="${CKPT:-$root/robomme/checkpoints/$config/$exp/$step/params}"  # CKPT: evaluate another config's params (e.g. the _shift1 counterfactual config on the _off checkpoint)
manifest="$root/robomme/metadata/PickXtimes/manifest.json"
# 09-15: the sidecar follows the label version of the config (official for the _off_ configs, v2 otherwise) unless
# SIDECAR overrides it (e.g. a shifted-history sidecar for the counterfactual oracle pass).
case "$config" in *_shift1_*) default_sidecar="$root/robomme/metadata/PickXtimes/subtasks_shift1.json" ;; *_off_*) default_sidecar="$root/robomme/metadata/PickXtimes/subtasks_official.json" ;; *) default_sidecar="$root/robomme/metadata/PickXtimes/subtasks_v2.json" ;; esac
sidecar="${SIDECAR:-$default_sidecar}"
out="$root/robomme/diagnostics/videos_${exp}_${step}${TAGSUF:-}"; mkdir -p "$out"
eps="${EPS:-0 2 3}"  # first easy / medium / hard episodes of the manifest (override with EPS=...)
run() { srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:"${GRES:-1}" env CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
[ -e "$ck" ] || { echo "missing params $ck" >> "$out/status.log"; exit 3; }
echo "battery start $(date +%m/%d\ %H:%M) host=$(hostname) job=$JOB gpu=$GPU ck=$ck modes=$modes eps=$eps" >> "$out/status.log"
for mode in $modes; do
  for ep in $eps; do
    tag=$(printf 'ep%02d_%s' "$ep" "$mode")${EXTRA_TAG:-}  # same tag as v5_heldout_video.py writes (ep%02d), so finished episodes are skipped
    [ -e "$out/$tag.json" ] && continue
    run .venv/bin/python scripts/v5_heldout_video.py --config-name "$config" --params "$ck" --episode-index "$ep" \
        --write-mode "$mode" --output-dir "$out" --manifest "$manifest" --sidecar "$sidecar" ${EXTRA:-} > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M) $(grep -o 'decision steps .*' "$out/${tag}_run.log" | tail -1 | cut -c1-160)" >> "$out/status.log"
  done
done
echo "all videos done $(date +%H:%M)" >> "$out/status.log"
