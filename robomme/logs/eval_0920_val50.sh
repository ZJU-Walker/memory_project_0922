#!/usr/bin/env bash
# PickXtimes 0920_v0 rollout eval on ALL 50 val episodes on the user's free 1 x H100 (job 17533970, iris-hgx-1; user
# 09-21 13:33 "run the eval on pickxtimes ... on all 50", 13:45 "do parallel if gpu memory permits, take full use of the gpu").
# Starts NLANES lanes (eval_0920_lane.sh: one model server + one simulator loop each) that share the episode queue, waits,
# then writes summary.json. More lanes can be added while it runs (another srun step: LANE=k bash eval_0920_lane.sh <step> <port>).
#   setsid nohup srun --jobid=17533970 --overlap --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:1 \
#       env NLANES=2 bash robomme/logs/eval_0920_val50.sh 2000 > robomme/logs/eval_0920_val50_2000.out 2>&1 &
set -u
STEP=${1:?step}; NLANES=${NLANES:-2}; PORT0=${PORT0:-18901}
ROOT=/iris/u/kewalk/memory_project_0920
OUT=${OUT:-$ROOT/robomme/replay/val50_0920_v0_$STEP}; mkdir -p "$OUT/logs"
export OUT HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date +%T)] $*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-?} step=$STEP lanes=$NLANES out=$OUT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
T0=$(date +%s); PIDS=()
for k in $(seq 0 $((NLANES - 1))); do
  LANE=$k bash "$ROOT/robomme/logs/eval_0920_lane.sh" "$STEP" $((PORT0 + k)) > "$OUT/logs/lane_$k.log" 2>&1 &
  PIDS+=($!); log "lane $k pid ${PIDS[-1]} port $((PORT0 + k))"
  sleep 20   # stagger the compiles
done
wait "${PIDS[@]}"
log "lanes finished after $(( $(date +%s) - T0 )) s"
$ROOT/openpi/.venv/bin/python - "$ROOT/robomme/rollouts" "$T0" "$OUT" <<'PY'
import json, sys, pathlib, time
root, t0, out = pathlib.Path(sys.argv[1]), float(sys.argv[2]), pathlib.Path(sys.argv[3])
rows = {}
for m in sorted(root.glob('*_PickXtimes_ep*_policy_*/manifest.json')):
    d = json.load(open(m))
    if float(d.get('started_at', 0)) < t0 - 5 or d.get('state') != 'complete':
        continue
    rows[d['episode']] = dict(episode=d['episode'], success=d.get('success'), status=d.get('status'), steps=d.get('steps'),
                              predictions=d.get('predictions'), goal=d.get('goal', ''), dir=m.parent.name,
                              secs=round(float(d.get('finished_at', time.time())) - float(d['started_at'])))
rows = [rows[k] for k in sorted(rows)]
json.dump(rows, open(out / 'summary.json', 'w'), indent=1)
for r in rows:
    print(f"ep{r['episode']:>3} {str(r['success']):5} {r['status']:8} steps={r['steps']} q={r['predictions']} {r['secs']}s | {r['goal'][:70]}")
print(f"SUCCESS {sum(r['success'] is True for r in rows)}/{len(rows)}")
PY
log "DONE step=$STEP" | tee "$OUT/DONE"
