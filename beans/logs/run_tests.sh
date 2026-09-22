#!/usr/bin/env bash
# CPU tests of the beans0922 tree as an srun step of $JOB (ON iris-hgx-1). Output: beans/logs/tests_cpu.out
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"; cd "$ROOT/openpi" || exit 2
srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MEMORY_PROJECT_ROOT="$ROOT" \
  .venv/bin/python -m pytest -q src/openpi/training/beans0922_test.py src/openpi/training/beans_0920_test.py src/openpi/training/robomme_0920_test.py -p no:cacheprovider > "$ROOT/beans/logs/tests_cpu.out" 2>&1
