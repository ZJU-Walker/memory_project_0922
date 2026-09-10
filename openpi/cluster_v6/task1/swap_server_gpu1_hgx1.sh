#!/usr/bin/env bash
# Run ON hgx-1: stop whatever B6 server runs on GPU 1 / port 8001 and serve checkpoint $1 there (threshold $WC, logging).
#   bash swap_server_gpu1_hgx1.sh 1400
step="$1"; WC="${WC:-0.6}"; PORT="${PORT:-8001}"
export HOME=/iris/u/kewalk
pids=$(ps -eo pid,cmd | grep -E "[s]erve_yam_memory.py --dir .*task1B6_20260909_r3/[0-9]+ .*--port $PORT|[s]run .*CUDA_VISIBLE_DEVICES=1 .*serve_yam_memory.py" | awk '{print $1}')
echo "stopping on port $PORT: $pids"; [ -n "$pids" ] && kill $pids 2>/dev/null; sleep 6
echo "left: $(ps -eo pid,cmd | grep -cE "[s]erve_yam_memory.py --dir .*--port $PORT")"
cd /iris/u/kewalk/memory_project_v6/openpi || exit 2
PYTHONDONTWRITEBYTECODE=1 MEM_FRACTION=0.9 JOB=17356154 GRES=2 GPU=1 CPUS=12 NO_PLACEHOLDER=1 SERVE_EXTRA="--num-steps 10 --write-conf $WC" \
  LOG=/iris/u/kewalk/memory_project_v6/v6/diagnostics/server_v6_b6_${step}_wc${WC/./}_hgx1.log \
  setsid nohup bash cluster_v6/serve_v6_job_v2.sh /iris/u/kewalk/memory_project_v6/v6/checkpoints/pi05_yam_mem_v6_task1B6/v6_task1B6_20260909_r3/$step pi05_yam_mem_v6_task1B6 $PORT > /dev/null 2>&1 < /dev/null &
echo "launched $step on port $PORT $(date +%H:%M:%S)"
