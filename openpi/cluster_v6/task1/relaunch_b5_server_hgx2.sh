#!/usr/bin/env bash
# Run ON iris-hgx-2: (re)launch the B5-500 robot server (12 CPUs, NUM_STEPS flow steps, default 10) if none is running.
export HOME=/iris/u/kewalk
ck=/iris/u/kewalk/memory_project_v6/v6/checkpoints/pi05_yam_mem_v6_task1B5/v6_task1B5_20260909_r1/500
n=$(ps -eo cmd | grep -c "^.venv/bin/python -u scripts/serve_yam_memory.py --dir /iris/u/kewalk/memory_project_v6/v6/checkpoints/pi05_yam_mem_v6_task1B5")
echo "running B5 servers: $n"
if [ "$n" = "0" ]; then
  cd /iris/u/kewalk/memory_project_v6/openpi || exit 2
  JOB=17329416 GRES=1 GPU=0 CPUS=12 NO_PLACEHOLDER=1 SERVE_EXTRA="--num-steps ${NUM_STEPS:-10}" LOG=/iris/u/kewalk/memory_project_v6/v6/diagnostics/server_v6_b5_500_20260909c.log \
    setsid nohup bash cluster_v6/serve_v6_job_v2.sh $ck pi05_yam_mem_v6_task1B5 8000 > /dev/null 2>&1 < /dev/null &
  echo "launched $(date +%H:%M:%S)"
fi
