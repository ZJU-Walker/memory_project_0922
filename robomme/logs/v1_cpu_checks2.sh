#!/usr/bin/env bash
# CPU checks of the redefined v1 (two cameras, horizon 30): unit tests, config check, one real loader batch. ON iris-hgx-1, CPU only.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
source cluster_robomme/env.sh >/dev/null 2>&1
find src/openpi -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
echo "== pytest $(date +%H:%M:%S)"; .venv/bin/python -m pytest -q src/openpi/training/robomme_0920_test.py src/openpi/models/pi0_v0920_test.py -p no:cacheprovider 2>&1 | tail -4
echo "== config check"; .venv/bin/python $ROOT/robomme/logs/config_0920_check.py pi05_robomme_0920_v1 pi05_robomme_0920_v1_probe pi05_robomme_0920_binfill_v1 2>&1 | grep -E "^pi05|model:|data:|RobommeInputs|inputs_spec|weight loader|longest" | cut -c1-260
echo "== loader probe v1_smoke (one batch) $(date +%H:%M:%S)"; srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 .venv/bin/python $ROOT/robomme/logs/probe_0920_data.py pi05_robomme_0920_v1_smoke 2>&1 | grep -E "images|actions|mask|prompt tokens|Traceback|Error" | head -12 | cut -c1-300
echo "== done $(date +%H:%M:%S)"
