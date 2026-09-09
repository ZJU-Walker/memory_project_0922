#!/usr/bin/env bash
# Run ON iris-hgx-2 (job 17329416): model-path bank recall probe on B2 checkpoints + the pointer beta value, to tell
# "bank lookup broken" from "closing decoder ignores the pointer" after the B2-1000 read collapse.
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
root=/iris/u/kewalk/memory_project_v6; cd $root/openpi || exit 2; source cluster_v6/env.sh >/dev/null 2>&1
out=$root/v6/diagnostics/bank_recall_probe; t1=$root/openpi/cluster_v6/task1
run() { srun --jobid=17329416 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 "$@"; }
for step in 1000 500; do
  ck=$root/v6/checkpoints/pi05_yam_mem_v6_task1B2/v6_task1B2_20260909_r1/$step/params; [ -e "$ck" ] || { echo "missing $ck"; continue; }
  run .venv/bin/python scripts/v6_bank_recall_probe.py --config-name pi05_yam_mem_v6_task1B2 --params "$ck" \
      --manifest $t1/task1v6_episode_manifest_v1.json --sidecar $t1/task1v6_v5_subtask_labels_v1.json \
      --decisions "open bin 1,open bin 2,open bin 3" --output $out/task1_B2_${step}.txt > $out/task1_B2_${step}.log 2>&1
  echo "B2 $step probe exit=$? $(date +%H:%M)"; tail -3 $out/task1_B2_${step}.txt
  run .venv/bin/python - "$ck" <<'PY' 2>&1 | grep -v Warning | tail -3
import sys, jax, numpy as np, orbax.checkpoint as ocp
tree = ocp.PyTreeCheckpointer().restore(sys.argv[1])
def walk(t, p=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from walk(v, f"{p}/{k}")
    else: yield p, t
for p, v in walk(tree):
    if "pointer" in p and ("beta" in p or "W_q" in p or "query" in p):
        a = np.asarray(v); print(p, a.shape, "beta=" if a.size == 1 else "norm=", float(a.ravel()[0]) if a.size == 1 else float(np.linalg.norm(a)))
PY
done
echo "probe done $(date +%H:%M)"
