#!/usr/bin/env bash
# Dev-episode battery for the phase-context ablation (cluster_v7/README.md §2) on the 2xH100 of job 17356154:
# the four ctx checkpoints (4999) x the development demos (demo11 demo19 demo37), raw-demo evaluator
# scripts/eval_yam_subtask_raw.py at stride 15 (2 Hz), the model feeding its OWN previous sentence (--prev-mode own,
# the deployment condition; batch 1) and, for the prev variants, the ground-truth previous sentence as the upper bound.
# Two configs per GPU, GPUs in parallel. Outputs v7/diagnostics/ctx_eval_<exp>_4999/{<demo>_<mode>_run.log,status.log,
# timelines.txt} (+ overlay mp4 / joint plots in scripts/eval_results/). Run ON iris-hgx-1:
#   nohup setsid bash cluster_v7/boba/run_ctx_evals_hgx1.sh > v7/logs/ctx_evals_hgx1.out 2>&1 &
set -u
JOB="${JOB:-17356154}"; STEP="${STEP:-4999}"; STRIDE="${STRIDE:-15}"; TAG="${TAG:-20260912_r1}"
root=/iris/u/kewalk/memory_project_v7
raw=/iris/u/kewalk/memory_project/data/boba_0911
prompt='make a boba tea, then a red bean tea'
DEMOS="${DEMOS:-demo11 demo19 demo37}"
cd "$root/openpi" || exit 2
source cluster_v7/env.sh >/dev/null 2>&1
export HOME=/iris/u/kewalk
one() {  # gpu config-suffix modes
  local gpu="$1" name="$2" modes="$3" cfg="pi05_yam_boba0911_$2" exp="$2_$TAG"
  local ck="$root/v7/checkpoints/$cfg/$exp/$STEP" out="$root/v7/diagnostics/ctx_eval_${exp}_${STEP}"
  mkdir -p "$out"
  [ -d "$ck/params" ] || { echo "no checkpoint at $ck/params $(date +%H:%M)" >> "$out/status.log"; return; }
  echo "ctx eval $exp ckpt-$STEP started $(date '+%m/%d %H:%M') gpu=$gpu modes=$modes demos: $DEMOS" >> "$out/status.log"
  for mode in $modes; do for demo in $DEMOS; do
    tag="${demo}_${mode}"; [ -e "$out/${tag}_run.log" ] && grep -q "^exact-match subtask:" "$out/${tag}_run.log" && continue
    srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=6 --gres=gpu:2 \
      env CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/eval_yam_subtask_raw.py \
        --config "$cfg" --ckpt-dir "$ck" --raw-demo "$raw/$demo" --prompt "$prompt" \
        --gt-labels "$raw/$demo/subtask_labels.json" --stride "$STRIDE" --batch-size 8 --prev-mode "$mode" \
        > "$out/${tag}_run.log" 2>&1
    echo "$tag exit=$? $(date +%H:%M) $(grep -hE '^exact-match subtask:|^sentence changes:' "$out/${tag}_run.log" | tr '\n' ' ' | cut -c1-200)" >> "$out/status.log"
    { echo "== $tag"; grep -hE "^(pred|gt) +timeline:|^exact-match subtask:|^sentence changes:" "$out/${tag}_run.log"; } >> "$out/timelines.txt" 2>/dev/null
  done; done
  echo "ctx eval $exp done $(date '+%m/%d %H:%M')" >> "$out/status.log"
}
( one 0 ctx_none own; one 0 ctx_state own ) &
( one 1 ctx_prev "own gt"; one 1 ctx_both "own gt" ) &
wait
echo "all ctx evals done $(date '+%m/%d %H:%M')"
