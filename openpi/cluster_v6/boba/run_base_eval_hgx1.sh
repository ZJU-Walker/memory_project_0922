#!/usr/bin/env bash
# Score the boba pi05+KI base checkpoint on the manifest's development episodes with the raw-demo evaluator
# (scripts/eval_yam_subtask_raw.py: reads the RAW demo folder, runs Pi0.sample_subtask_and_actions per frame ->
# subtask-overlay mp4 + predicted-vs-teleop joint plots in scripts/eval_results/, exact-match subtask accuracy vs
# the demo's own subtask_labels.json on stdout). Run ON iris-hgx-1:
#   [JOB=17356154] [STEP=9999] [EXP=pi05_boba0911_base_rtc15_20260912_r1] [GPU=0] [DEMOS="demo11 demo19 demo37"] \
#     nohup setsid bash cluster_v6/boba/run_base_eval_hgx1.sh > v6/logs/base_eval_boba.out 2>&1 &
set -u
JOB="${JOB:-17356154}"; STEP="${STEP:-9999}"; EXP="${EXP:-pi05_boba0911_base_rtc15_20260912_r1}"
CONFIG=pi05_yam_boba0911_base; GPU="${GPU:-0}"; STRIDE="${STRIDE:-15}"; BATCH="${BATCH:-8}"
root=/iris/u/kewalk/memory_project_v6
raw=/iris/u/kewalk/memory_project/data/boba_0911
prompt='make a boba tea, then a red bean tea'
# manifest 0911_boba_episode_manifest_v1.json split=development: demo11 demo19 demo37 (final_test: demo13 demo41 demo59)
DEMOS="${DEMOS:-demo11 demo19 demo37}"
cd "$root/openpi" || exit 2
source cluster_v6/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
ck="$root/v6/checkpoints/$CONFIG/$EXP/$STEP"
[ -d "$ck/params" ] || { echo "no checkpoint at $ck/params"; exit 2; }
out="$root/v6/diagnostics/base_eval_${EXP}_${STEP}"; mkdir -p "$out"
echo "base eval $EXP ckpt-$STEP started $(date '+%m/%d %H:%M') gpu=$GPU demos: $DEMOS" >> "$out/status.log"
for demo in $DEMOS; do
  [ -d "$raw/$demo" ] || { echo "$demo MISSING $(date +%H:%M)" >> "$out/status.log"; continue; }
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 \
    env CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval_yam_subtask_raw.py \
      --config "$CONFIG" --ckpt-dir "$ck" --raw-demo "$raw/$demo" --prompt "$prompt" \
      --gt-labels "$raw/$demo/subtask_labels.json" --stride "$STRIDE" --batch-size "$BATCH" \
      > "$out/${demo}_run.log" 2>&1
  echo "$demo exit=$? $(date +%H:%M)" >> "$out/status.log"
  grep -hE "^(pred|gt) +timeline:|^exact-match subtask:" "$out/${demo}_run.log" >> "$out/timelines.txt" 2>/dev/null
done
echo "base eval done $(date '+%m/%d %H:%M'); artifacts in scripts/eval_results/" >> "$out/status.log"
