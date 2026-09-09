#!/usr/bin/env bash
# Run ON iris-hgx-2: (re)start the generic v2 gate for A2 -> B2 (script file: an inline pkill+relaunch matches its own
# command line and kills itself). Env passthrough: PARALLEL (default 1 on the cleared H200), MIN_RECALL, NEED_MIB.
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs
pkill -u kewalk -f "gate_generic_v2_hgx[2]"; sleep 2
echo "old gate alive: $(pgrep -f "gate_generic_v2_hgx[2]" | wc -l)"
cd $root/openpi
echo "$(date '+%m/%d %H:%M') [hgx-2] gate v2 (re)started, PARALLEL=${PARALLEL:-1}" >> $logs/gate_task1_hgx2.log
PARALLEL=${PARALLEL:-1} ACFG=pi05_yam_mem_v6_task1A2 AEXP=v6_task1A2_20260909_r1 BCFG=pi05_yam_mem_v6_task1B2 BEXP=v6_task1B2_20260909_r1 ENVVAR=OPENPI_V6_TASK1_A2_PARAMS \
  setsid nohup bash cluster_v6/task1/gate_generic_v2_hgx2.sh > $logs/gate_generic_v2_A2.out 2>&1 < /dev/null & disown
sleep 2; echo "gate procs now: $(pgrep -f "gate_generic_v2_hgx[2]" | wc -l)"; tail -1 $logs/gate_task1_hgx2.log | cut -c1-200
