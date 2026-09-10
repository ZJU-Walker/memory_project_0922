#!/usr/bin/env bash
# GPU sentinel for the shared test H200 (job 17329416, iris-hgx-2). User 2026-09-09 02:44: "while you are not using the
# H200 make it busy". Copy of cluster_v5/gpu_sentinel_job.sh with (a) a TUNED placeholder that preallocates only half the
# card (MEM_FRAC 0.5 = ~72 GB, matmul duty ~0.9) so a battery/probe process (~20 GB) can never OOM in the 20 s race
# before the sentinel kills the placeholder, and (b) the v6 runners in the real-work list. Real work always wins:
# any python of THIS job that is neither the placeholder nor the train_hs.py keep-alive counts as real work.
# Never touches any other job (job id read from /proc/<pid>/environ; marker carries the job id).
#   JOB=17329416 setsid nohup bash cluster_v6/gpu_sentinel_h200_v2.sh > /dev/null 2>&1 < /dev/null &   (run ON iris-hgx-2)
export HOME=/iris/u/kewalk
JOB="${JOB:-17329416}"
LOGDIR=/iris/u/kewalk/memory_project_v6/v6/logs; mkdir -p $LOGDIR; LOG=$LOGDIR/sentinel_h200.log
MARKER="gpu_placeholder_marker_${JOB}"
REAL="jobid=${JOB} .*(trai[n]|scripts/v[56]_[a-z0-9_]*\.p[y])|scripts/v[56]_[a-z0-9_]*\.p[y] |v5_heldout_vide[o]|v6_bank_recall_prob[e]|v6_model_bank_prob[e]|run_task1_evals_hgx[1]|serve_yam_memor[y]"
job_busy() {
  # v2 (2026-09-09 19:25): decide from the processes that actually hold the card (nvidia-smi compute apps), not from
  # SLURM_JOB_ID in /proc environ — launcher shells and srun clients of OTHER jobs started from an ssh login carry this
  # job's id (cgroup adoption) and kept the v1 sentinel "busy" forever. Keep-alive and placeholder do not count.
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE "train_hs\.py|gpu_placeholder_marker" && continue
    grep -qz "^GPU_PLACEHOLDER=" /proc/$p/environ 2>/dev/null && continue
    return 0
  done
  return 1
}
echo "$(date '+%m/%d %H:%M') sentinel v2 up for job $JOB on $(hostname) pid=$$ (tuned placeholder MEM_FRAC 0.5; busy = GPU compute apps or v6 runners)" >> $LOG
while true; do
  busy=0; pgrep -af "$REAL" | grep -v "$MARKER" | grep -q . && busy=1
  [ $busy = 0 ] && job_busy && busy=1
  ph=$(pgrep -f "^srun .*$MARKER" | tr '\n' ' ')
  any=$(pgrep -f "$MARKER" | tr '\n' ' ')
  if [ $busy = 1 ] && [ -n "$ph" ]; then
    echo "$(date '+%m/%d %H:%M:%S') real work detected while the placeholder runs -> killing placeholder srun pids $ph" >> $LOG
    kill $ph 2>/dev/null
  elif [ $busy = 0 ] && [ -z "$any" ]; then
    echo "$(date '+%m/%d %H:%M:%S') job $JOB idle -> launching the tuned placeholder" >> $LOG
    JOB=$JOB GPU=0 MEM_FRAC=0.5 DUTY=0.92 HOLD_GB=32 bash /iris/u/kewalk/memory_project_v5/openpi/cluster_v5/gpu_placeholder_tuned.sh >> $LOG 2>&1
  fi
  sleep 20
done
