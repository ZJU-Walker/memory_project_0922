#!/bin/bash
# Self-contained battery job for the fresh-line B6 checkpoints on ONE iris-partition GPU (user 2026-09-10 03:30:
# "use sc slurm to request a gpu l40 or a40 ... use iris not iris-hi ... and use this to run battery").
#   ssh sc sbatch --gres=gpu:l40s:1 cluster_v6/task1/battery_B6_sbatch.sh   (or --gres=gpu:a40:1)
# For each step in STEPS (default 200 400 ... 2000) it waits until the checkpoint is finalized, runs the battery modes
# one after another on the allocated GPU (48 GB: one mode at a time), then writes verdict.txt. Exits after the last step.
#SBATCH --job-name=v6_task1B6_battery
#SBATCH --output=/iris/u/kewalk/memory_project_v6/v6/logs/battery_sbatch-%j.out
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=iris
set -u
EXP="${EXP:-v6_task1B6_20260909_r3}"; CFG="${CFG:-pi05_yam_mem_v6_task1B6}"; MODES="${MODES:-oracle_evidence self}"
STEPS="${STEPS:-200 400 600 800 1000 1200 1400 1600 1800 2000}"
root=/iris/u/kewalk/memory_project_v6; cv6=$root/openpi/cluster_v6
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1
export SIDECAR=$cv6/task1/task1v6_v5_subtask_labels_v1tailgo.json MANIFEST=$cv6/task1/task1v6_episode_manifest_v1tailgo.json
cd "$root/openpi" || exit 2
echo "==== battery job $SLURM_JOB_ID on $SLURMD_NODENAME $(date) ===="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in $STEPS; do
  out=$root/v6/diagnostics/videos_${EXP}_${step}; mkdir -p "$out"
  while ! { [ -e "$root/v6/checkpoints/$CFG/$EXP/$step/params" ] && grep -q "\[step=$step\] CheckpointManager Save Finalize is done on all hosts" "$root/v6/logs/train_$EXP.log"; }; do
    tail -1 "$root/v6/logs/train_${EXP}_status.log" | grep -q "^exit=" && ! [ -e "$root/v6/checkpoints/$CFG/$EXP/$step/params" ] && { echo "training exited before step $step; stopping"; exit 0; }
    sleep 60
  done
  echo "step $step: battery start $(date +%m/%d\ %H:%M) job=$SLURM_JOB_ID node=$SLURMD_NODENAME" | tee -a "$out/status.log"
  for m in $MODES; do MODES=$m JOB=$SLURM_JOB_ID GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$CFG" "$EXP" "$step"; done
  .venv/bin/python scripts/task1_battery_verdict.py --sidecar "$SIDECAR" "$out" > "$out/verdict.txt" 2>&1
  echo "verdict written $(date +%m/%d\ %H:%M)" | tee -a "$out/status.log"; cat "$out/verdict.txt"
done
echo "all steps done $(date)"
