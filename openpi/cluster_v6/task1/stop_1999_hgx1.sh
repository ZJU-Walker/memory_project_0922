#!/usr/bin/env bash
# Run ON hgx-1: stop the B6-1999 server (waiter script, srun step and python) on GPU 0 / port 8000. Nothing else.
pids=$(ps -eo pid,cmd | grep -E "[s]erve_1999_hgx1.sh|[s]erve_yam_memory.py --dir .*task1B6_20260909_r3/1999|[s]run .*serve_yam_memory.py --dir .*task1B6_20260909_r3/1999" | awk '{print $1}')
echo "stopping: $pids"; [ -n "$pids" ] && kill $pids 2>/dev/null; sleep 5
echo "left: $(ps -eo pid,cmd | grep -cE "[s]erve_yam_memory.py --dir .*1999")"
