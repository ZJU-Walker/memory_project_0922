#!/usr/bin/env bash
# Run ON iris-hgx-2 (job 17329416): decision-phrasing read probe on a v6 checkpoint (see scripts/v6_context_phrasing_probe.py).
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
root=/iris/u/kewalk/memory_project_v6; cd $root/openpi || exit 2; source cluster_v6/env.sh >/dev/null 2>&1
t1=$root/openpi/cluster_v6/task1; out=$root/v6/diagnostics/bank_recall_probe
srun --jobid=17329416 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 env CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/v6_context_phrasing_probe.py --config-name "${CFG:-pi05_yam_mem_v6_task1B3}" --params "${PARAMS:?}" \
  --manifest $t1/task1v6_episode_manifest_v1lead30.json --sidecar $t1/task1v6_v5_subtask_labels_v1lead30.json --output $out/phrasing_${TAG:-x}.txt > $out/phrasing_${TAG:-x}.log 2>&1
echo "probe exit=$? $(date +%H:%M)"; tail -8 $out/phrasing_${TAG:-x}.txt
