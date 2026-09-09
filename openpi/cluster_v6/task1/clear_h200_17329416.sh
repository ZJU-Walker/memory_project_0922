#!/usr/bin/env bash
# Run ON iris-hgx-2. User 2026-09-09 01:32: "for your H200 you can just stop the other servers but keep 1gb alive".
# Stops every GPU process that belongs to job 17329416 EXCEPT the train_hs.py keep-alive; processes of any other job
# (e.g. 17286852) are never touched (SLURM_JOB_ID is read from /proc/<pid>/environ). Script file: no pgrep self-match.
export HOME=/iris/u/kewalk
KEEP_JOB=17329416
echo "--- GPU processes seen from this cgroup ($(grep -o 'job_[0-9]*' /proc/self/cgroup | head -1)):"
victims=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  job=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^SLURM_JOB_ID=' | cut -d= -f2)
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' -v p=$p '$1==p {print $2}' | head -1)
  case "$cmd" in
    *train_hs.py*) echo "  keep  pid $p job=$job $mem  $cmd";;
    *) if [ "$job" = "$KEEP_JOB" ]; then echo "  STOP  pid $p job=$job $mem  $cmd"; victims="$victims $p"; else echo "  skip  pid $p job=${job:-?} $mem (not job $KEEP_JOB)  $cmd"; fi;;
  esac
done
if [ -n "$victims" ]; then
  kill -TERM $victims 2>/dev/null; sleep 20
  left=""; for p in $victims; do [ -d /proc/$p ] && left="$left $p"; done
  [ -n "$left" ] && { echo "  SIGKILL$left"; kill -KILL $left 2>/dev/null; sleep 5; }
fi
echo "--- after:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
