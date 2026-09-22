#!/usr/bin/env bash
# CPU checks of the BinFill configs against the released-only BinFill100 dataset (run ON iris-hgx-1, no GPU): config
# construction + one real loader batch (prompt lengths, history masks, window shapes). Waits for the BinFill100 spec.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OPENPI_0920_BINFILL_DATASET=${OPENPI_0920_BINFILL_DATASET:-BinFill100}
source cluster_robomme/env.sh >/dev/null 2>&1
spec=$ROOT/robomme/metadata/$OPENPI_0920_BINFILL_DATASET/prepared_official.json
n=0; until [ -f "$spec" ] || [ $n -ge 240 ]; do sleep 15; n=$((n+1)); done
[ -f "$spec" ] || { echo "no spec after $((n*15))s"; exit 1; }
echo "== spec $(date +%H:%M:%S)"; .venv/bin/python -c "import json,sys; d=json.load(open('$spec')); print({k: d[k] for k in ('task','episodes','frames','max_segments_per_episode','max_sentence_tokens','lerobot_dataset','episodes_by_source')}); print('sentences', d['sentences'])"
echo "== config check"; .venv/bin/python $ROOT/robomme/logs/config_0920_check.py pi05_robomme_0920_binfill_v0 pi05_robomme_0920_binfill_v0_smoke 2>&1 | grep -v "^WARNING\|^I0\|^W0" | cut -c1-400
echo "== PickXtimes config unchanged?"; .venv/bin/python $ROOT/robomme/logs/config_0920_check.py pi05_robomme_0920_v0 2>&1 | grep -E "steps|data:|sets:" | cut -c1-300
echo "== loader probe (one batch)"; srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env JAX_PLATFORMS=cpu OPENPI_0920_BINFILL_DATASET=$OPENPI_0920_BINFILL_DATASET .venv/bin/python $ROOT/robomme/logs/probe_0920_data.py pi05_robomme_0920_binfill_v0_smoke 2>&1 | grep -v "^WARNING\|^I0\|^W0" | cut -c1-400
echo "== done $(date +%H:%M:%S)"
