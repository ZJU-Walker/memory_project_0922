#!/usr/bin/env bash
# CPU checks of the 0920 v1 (no past frames) configs: unit tests, config construction, one real loader batch. Run ON iris-hgx-1
# inside our 2xH100 job (CPU only, JAX_PLATFORMS=cpu). 09-21 14:40.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
source cluster_robomme/env.sh >/dev/null 2>&1
find src/openpi/training -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
echo "== pytest $(date +%H:%M:%S)"; .venv/bin/python -m pytest -q src/openpi/training/robomme_0920_test.py -p no:cacheprovider 2>&1 | tail -6
echo "== config check v1"; .venv/bin/python $ROOT/robomme/logs/config_0920_check.py pi05_robomme_0920_v1 pi05_robomme_0920_v1_smoke pi05_robomme_0920_binfill_v1 2>&1 | grep -v "^WARNING\|^I0\|^W0" | cut -c1-300
echo "== v0 unchanged?"; .venv/bin/python $ROOT/robomme/logs/config_0920_check.py pi05_robomme_0920_v0 pi05_robomme_0920_binfill_v0 2>&1 | grep -E "steps|hist|data:|RobommeInputs|inputs_spec" | cut -c1-300
echo "== loader probe v1_smoke (one batch) $(date +%H:%M:%S)"; srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 .venv/bin/python $ROOT/robomme/logs/probe_0920_data.py pi05_robomme_0920_v1_smoke 2>&1 | grep -v "^WARNING\|^I0\|^W0" | cut -c1-300
echo "== done $(date +%H:%M:%S)"
