#!/usr/bin/env bash
# Run ON iris-hgx-2 (job 17329416): print the v6 pointer beta (and any pointer leaves) of the given param dirs.
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
root=/iris/u/kewalk/memory_project_v6; cd $root/openpi || exit 2; source cluster_v6/env.sh >/dev/null 2>&1
srun --jobid=17329416 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu .venv/bin/python - "$@" <<'PY' 2>&1 | grep -v "Warning\|warn"
import sys, numpy as np
from openpi.models import model as _m
for path in sys.argv[1:]:
    params = _m.restore_params(path, restore_type=np.ndarray)
    def walk(t, p=""):
        if isinstance(t, dict):
            for k, v in t.items(): yield from walk(v, f"{p}/{k}")
        else: yield p, t
    for p, v in walk(params):
        if "pointer" in p:
            a = np.asarray(v, dtype=np.float32)
            print(path.split("/v6/")[-1], p, a.shape, f"value={float(a.ravel()[0]):.4f}" if a.size == 1 else f"fro_norm={float(np.linalg.norm(a)):.4f}")
PY
echo "peek done $(date +%H:%M)"
