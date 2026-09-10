#!/usr/bin/env bash
# Battery + verdict for ONE B6 (fresh line, tailgo labels) checkpoint on the H200 of job 17329416. Run ON iris-hgx-2:
#   [EXP=v6_task1B6_20260909_r3] [CFG=pi05_yam_mem_v6_task1B6] [MODES="oracle_evidence self"] setsid nohup bash cluster_v6/task1/battery_B6_step_hgx2.sh <step> &
# Waits for the checkpoint to be finalized, runs the modes in parallel (~24 GB each), then writes verdict.txt.
set -u
step="$1"; EXP="${EXP:-v6_task1B6_20260909_r3}"; CFG="${CFG:-pi05_yam_mem_v6_task1B6}"; MODES="${MODES:-oracle_evidence self}"
root=/iris/u/kewalk/memory_project_v6; cv6=$root/openpi/cluster_v6; out=$root/v6/diagnostics/videos_${EXP}_${step}
export HOME=/iris/u/kewalk SIDECAR=$cv6/task1/task1v6_v5_subtask_labels_v1tailgo.json MANIFEST=$cv6/task1/task1v6_episode_manifest_v1tailgo.json
mkdir -p "$out"; cd "$root/openpi" || exit 2
while ! { [ -e "$root/v6/checkpoints/$CFG/$EXP/$step/params" ] && grep -q "\[step=$step\] CheckpointManager Save Finalize is done on all hosts" "$root/v6/logs/train_$EXP.log"; }; do sleep 30; done
echo "runner $(date +%m/%d\ %H:%M) host=$(hostname) step=$step modes=$MODES" >> "$out/status.log"
for m in $MODES; do MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$CFG" "$EXP" "$step" & done; wait
.venv/bin/python scripts/task1_battery_verdict.py --sidecar "$SIDECAR" "$out" > "$out/verdict.txt" 2>&1
echo "verdict written $(date +%m/%d\ %H:%M)" >> "$out/status.log"
