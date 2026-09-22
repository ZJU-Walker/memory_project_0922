#!/usr/bin/env bash
# One eval lane for the PickXtimes 0920_v0 val sweep: its own model server on PORT + one simulator loop that claims episodes
# from the shared queue $OUT/claims (mkdir = atomic claim, so any number of lanes can run at once and lanes can be added
# while the sweep runs). Memory tick 20 control steps (training stride), 20 of the 40 planned actions executed per tick,
# official 20 Hz control, own notes. Each finished episode is exported for the sweep page immediately.
#   LANE=<k> bash robomme/logs/eval_0920_lane.sh <step> <port>      (inside the eval job's srun step, CUDA_VISIBLE_DEVICES set)
# Only this lane's own server is ever stopped (by pid).
set -u
STEP=${1:?step}; PORT=${2:?port}; LANE=${LANE:-$PORT}
ROOT=/iris/u/kewalk/memory_project_0920; EVAL=$ROOT/openpi/cluster_robomme/eval
CONFIG=pi05_robomme_0920_v0; EXP=robomme_0920_v0
CKPT=$ROOT/robomme/checkpoints/$CONFIG/$EXP/$STEP
TAG=${TAG:-}   # diagnostic variants get their own output dir and name prefix, e.g. TAG=nohist EXTRA_ARGS="--mask-history"
OUT=${OUT:-$ROOT/robomme/replay/val50_0920_v0_${STEP}${TAG:+_$TAG}}; mkdir -p "$OUT/logs" "$OUT/claims"
TICK=${TICK:-20}; EXEC=${EXEC:-20}; HORIZON=${HORIZON:-40}; NEPS=${NEPS:-50}; EXTRA_ARGS=${EXTRA_ARGS:-}
EPS=${EPS:-$(seq 0 $((NEPS - 1)) | tr "\n" " ")}   # episode queue (space separated)
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_ALLOCATOR=platform   # several servers + simulators share one card: no grow-only pool
log() { echo "[$(date +%T)] [lane $LANE] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-?} step=$STEP port=$PORT out=$OUT tick=$TICK exec=$EXEC"
[ -f "$CKPT/_CHECKPOINT_METADATA" ] && [ -d "$CKPT/params" ] || { log "no finalized checkpoint $CKPT"; exit 2; }
find "$ROOT/openpi/scripts" "$EVAL" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
T0=$(date +%s)
OPENPI_JAX_CACHE_DIR=$ROOT/robomme/cache/eval_jax_h100_lane$LANE setsid bash "$EVAL/serve.sh" --task PickXtimes --stage B --checkpoint "$CKPT" --port "$PORT" \
  > "$OUT/logs/serve_$PORT.log" 2>&1 < /dev/null &
SPID=$!
log "server pid $SPID"
for i in $(seq 1 600); do
  kill -0 "$SPID" 2>/dev/null || { log "server died"; tail -30 "$OUT/logs/serve_$PORT.log"; exit 3; }
  python3 -c "import urllib.request,sys; open(sys.argv[2],'wb').write(urllib.request.urlopen(sys.argv[1],timeout=3).read())" "http://127.0.0.1:$PORT/metadata" "$OUT/logs/metadata_$PORT.json" 2>/dev/null && break
  sleep 3
done
[ -s "$OUT/logs/metadata_$PORT.json" ] || { log "server never came up"; kill "$SPID" 2>/dev/null; exit 4; }
python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print('metadata', {k:m.get(k) for k in ('training_config','trained_update','memory_stride_frames','image_history_frames','image_history_stride','model_backend','ground_truth_input')})" "$OUT/logs/metadata_$PORT.json"
log "server up after $(( $(date +%s) - T0 )) s; gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
while true; do
  ep=""
  for e in $EPS; do
    if mkdir "$OUT/claims/ep$(printf %02d $e)" 2>/dev/null; then ep=$e; echo "lane $LANE $(date +%T)" > "$OUT/claims/ep$(printf %02d $e)/CLAIM"; break; fi
  done
  [ -n "$ep" ] || break
  t0=$(date +%s)
  bash "$EVAL/run.sh" --task PickXtimes --split val --episodes "$ep" --chunk-size $TICK --execute-horizon $EXEC --max-steps 1300 --seed 7 \
    --policy-url "http://127.0.0.1:$PORT" $EXTRA_ARGS > "$OUT/logs/ep$(printf %02d $ep).log" 2>&1
  rc=$?
  d=$($ROOT/openpi/.venv/bin/python - "$ROOT/robomme/rollouts" "$t0" "$ep" <<'PY'
import json, pathlib, sys
root, t0, ep = pathlib.Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
best = None
for m in root.glob("*_PickXtimes_ep%03d_policy_*/manifest.json" % ep):
    d = json.load(open(m))
    if d.get("state") == "complete" and float(d.get("started_at", 0)) >= t0 - 5:
        best = m.parent if best is None or d["started_at"] > json.load(open(best / "manifest.json"))["started_at"] else best
print(best or "")
PY
)
  if [ -n "$d" ]; then
    PYTHONPATH=$EVAL $ROOT/openpi/.venv/bin/python "$EVAL/rollout_replay_export.py" "$d" --out "$OUT" --horizon $HORIZON --name "v0_${STEP}${TAG:+_$TAG}_ep$(printf %02d $ep)" 2>&1 | tail -1
    python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print('  ep%02d %s steps=%s queries=%s | %s' % (m['episode'], m.get('status'), m.get('steps'), m.get('predictions'), (m.get('goal') or '')[:70]))" "$d/manifest.json"
    echo "$d" > "$OUT/claims/ep$(printf %02d $ep)/DONE"
  else
    log "ep $ep: no complete rollout (rc=$rc)"; tail -3 "$OUT/logs/ep$(printf %02d $ep).log"; echo "rc=$rc" > "$OUT/claims/ep$(printf %02d $ep)/FAILED"
  fi
  log "ep $ep done rc=$rc ($(( $(date +%s) - t0 )) s); gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
done
kill -TERM "$SPID" 2>/dev/null; sleep 5; kill -KILL "$SPID" 2>/dev/null   # this lane's own server only
log "queue empty, lane DONE after $(( $(date +%s) - T0 )) s" | tee "$OUT/logs/lane_${LANE}_DONE"
