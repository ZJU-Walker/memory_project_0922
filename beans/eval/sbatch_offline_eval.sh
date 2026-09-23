#!/bin/bash
#SBATCH --job-name=beans0922_eval
#SBATCH --requeue
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/iris/u/kewalk/memory_project_beans0922/beans/eval/slurm_%j.out
# Offline evaluation of beans0922 checkpoints on two L40S cards of the iris partition (user 09-22 16:43: "take 1 or 2 l40s and
# run offline eval there, use iris partition"). Per checkpoint in CKPTS: GPU 0 runs the count-flip battery
# (scripts/v5_count_flip_eval.py: go-count accuracy with true notes / follow rate under shifted notes / accuracy with an empty
# bank, 96 development windows), GPU 1 the held-out video probe (beans/eval/run_heldout_videos.sh, own + label notes, 6 dev
# episodes). The in-tree HF datasets cache is a symlink to node-local /scr: its target is created on this node and the arrow
# cache is rebuilt once per node (~40 min); the dataset itself is read from the tree (NFS).
#   sbatch --export=ALL,CFG=pi05_yam_beans0922_v1,EXP=beans0922_v1,CKPTS="2000 3000" beans/eval/sbatch_offline_eval.sh
# Knobs: TOOLS="battery videos" (one or both), BATTERY_GPU=0, VIDEO_GPU=1 (set both to 0 with --gres=gpu:l40s:1: the tools then run one
# after the other on the single card). Preflight: every card the job got must initialise CUDA (17:12 09-22: iris9's second card had an
# uncorrectable ECC error and every video run died with "no supported devices found for platform CUDA"); a bad card -> exit 4,
# resubmit with --exclude=<node>.
set -u
ROOT=${MEMORY_PROJECT_ROOT:-/iris/u/kewalk/memory_project_beans0922}
CFG=${CFG:-pi05_yam_beans0922_v1}; EXP=${EXP:-beans0922_v1}; CKPTS=${CKPTS:-2000}
TOOLS=${TOOLS:-"battery videos"}; BATTERY_GPU=${BATTERY_GPU:-0}; VIDEO_GPU=${VIDEO_GPU:-1}
FLIP_TOKENS=${FLIP_TOKENS:-sentence}  # battery CE over the sentence tokens only (all = the historical whole-buffer measure)
cd "$ROOT/openpi" || exit 2
export MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/iris/u/kewalk
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.95}
[ -f cluster_robomme/env.sh ] && source cluster_robomme/env.sh >/dev/null 2>&1
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets" HF_LEROBOT_HOME="$ROOT/data/lerobot"
export OPENPI_DATA_HOME="$ROOT/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax"
target=$(readlink -f "$HF_DATASETS_CACHE" 2>/dev/null || readlink "$HF_DATASETS_CACHE")
mkdir -p "$target" || { echo "cannot create the node-local datasets cache $target"; exit 3; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?} cache=$target cfg=$CFG exp=$EXP ckpts='$CKPTS' $(date)"
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
EVAL="$ROOT/beans/eval"
# preflight: each card must come up in JAX (a card with an uncorrectable ECC error shows as "no supported devices")
for g in $(echo "${BATTERY_GPU} ${VIDEO_GPU}" | tr ' ' '\n' | sort -u); do
  if ! CUDA_VISIBLE_DEVICES=$g timeout 300 .venv/bin/python -c "import jax; d=jax.devices(); assert d and d[0].platform=='gpu', d; print('card', $g, 'ok:', d[0].device_kind)"; then
    echo "card $g failed to initialise on $(hostname) -- bad GPU? exit 4 (resubmit with --exclude=$(hostname -s))"; nvidia-smi -i $g -q -d ECC | grep -iA2 "uncorrectable" | head -6; exit 4
  fi
done
run_battery() {  # $1 = step; batch 2 on a 48 GB card, batch 1 on OOM
  local step=$1 out="$EVAL/flip_${EXP}_${step}_${FLIP_TOKENS}" log="$EVAL/flip_${EXP}_${step}_${FLIP_TOKENS}.log" rc=1
  [ -e "$out/count_flip_eval.json" ] && { echo "battery $step: exists"; return 0; }
  for bs in ${FLIP_BATCH:-2} 1; do
    rm -rf "$out"
    CUDA_VISIBLE_DEVICES=$BATTERY_GPU .venv/bin/python scripts/v5_count_flip_eval.py --config-name "$CFG" \
      --params "$ROOT/beans/checkpoints/$CFG/$EXP/$step/params" --split development --batches $((96 / bs)) --batch-size $bs --tokens "$FLIP_TOKENS" \
      --output-dir "$out" > "$log" 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    grep -q RESOURCE_EXHAUSTED "$log" || break
    echo "battery $step: batch $bs ran out of memory, retrying smaller"
  done
  echo "battery $step exit=$rc $(date +%H:%M)"
  [ -e "$out/count_flip_eval.json" ] && python3 -c "
import json,sys; r=json.load(open('$out/count_flip_eval.json'))
print('battery $step:', {k: r.get(k) for k in ('normal_count_accuracy','flip_follows_content_rate','blank_count_accuracy','go_steps','windows') if k in r})"
  return $rc
}
run_videos() {  # $1 = step
  GPU=$VIDEO_GPU MODES="${MODES:-self oracle}" bash "$EVAL/run_heldout_videos.sh" "$CFG" "$EXP" "$1"
  echo "videos $1 done $(date +%H:%M): $(grep -c 'exit=0' "$EVAL/videos_${EXP}_$1/status.log" 2>/dev/null) runs ok"
}
for step in $CKPTS; do
  [ -d "$ROOT/beans/checkpoints/$CFG/$EXP/$step/params" ] || { echo "no checkpoint $step, skipping"; continue; }
  if [ "$BATTERY_GPU" != "$VIDEO_GPU" ]; then  # two cards: both tools at once
    case " $TOOLS " in *" battery "*) run_battery "$step" & pid_b=$!;; *) pid_b=;; esac
    case " $TOOLS " in *" videos "*) run_videos "$step" & pid_v=$!;; *) pid_v=;; esac
    [ -n "$pid_b" ] && wait $pid_b; [ -n "$pid_v" ] && wait $pid_v
  else  # one card: one after the other
    case " $TOOLS " in *" battery "*) run_battery "$step";; esac
    case " $TOOLS " in *" videos "*) run_videos "$step";; esac
  fi
done
echo "all done $(date)"
