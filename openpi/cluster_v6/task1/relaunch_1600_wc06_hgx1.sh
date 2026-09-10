#!/usr/bin/env bash
# Run ON hgx-1. (Re)launch the B6-1600 server on GPU 1 / port 8001 with per-step logging and write threshold 0.6.
export HOME=/iris/u/kewalk
n=$(ps -eo pid,cmd | grep -c "[s]erve_yam_memory.py --dir .*task1B6_20260909_r3/1600")
echo "running 1600 servers: $n"
if [ "$n" = "0" ]; then
  cd /iris/u/kewalk/memory_project_v6/openpi || exit 2
  PYTHONDONTWRITEBYTECODE=1 MEM_FRACTION=0.9 JOB=17356154 GRES=2 GPU=1 CPUS=12 NO_PLACEHOLDER=1 SERVE_EXTRA="--num-steps 10 --write-conf 0.6" \
    LOG=/iris/u/kewalk/memory_project_v6/v6/diagnostics/server_v6_b6_1600_wc06_hgx1.log \
    setsid nohup bash cluster_v6/serve_v6_job_v2.sh /iris/u/kewalk/memory_project_v6/v6/checkpoints/pi05_yam_mem_v6_task1B6/v6_task1B6_20260909_r3/1600 pi05_yam_mem_v6_task1B6 8001 > /dev/null 2>&1 < /dev/null &
  echo "launched $(date +%H:%M:%S)"
fi
