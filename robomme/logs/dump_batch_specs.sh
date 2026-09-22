#!/usr/bin/env bash
# CPU: one 40-tick batch (batch 8) of the v1 and v0 probe configs -> pickles for the AOT memory analysis. ON iris-hgx-1.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
source cluster_robomme/env.sh >/dev/null 2>&1
for c in pi05_robomme_0920_v1_probe pi05_robomme_0920_v0_probe; do
  echo "== $c $(date +%H:%M:%S)"
  srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=24 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python $ROOT/robomme/logs/dump_batch_spec.py $c $ROOT/robomme/logs/batch_spec_$c.pkl 2>&1 | grep -v "^WARNING\|^I0\|^W0\|UserWarning\|warnings.warn" | cut -c1-200
done
echo "== done $(date +%H:%M:%S)"
