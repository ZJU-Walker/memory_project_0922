#!/usr/bin/env bash
# Run ON iris-hgx-2 (script file on purpose): SIGSTOP the live gate v2 (pid $1) so the B2-500 battery does not collide
# with the two freeze_decision batteries on the H200; SIGCONT it when no freezedec process is left (or after 100 min).
export HOME=/iris/u/kewalk
gate="$1"; log=/iris/u/kewalk/memory_project_v6/v6/logs/gate_task1_hgx2.log
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
kill -STOP "$gate" && say "gate v2 (pid $gate) PAUSED while the freeze_decision batteries use the H200"
t=0
while pgrep -u kewalk -f "output-dir [^ ]*freezede[c]_" >/dev/null && [ $t -lt 6000 ]; do sleep 30; t=$((t+30)); done
kill -CONT "$gate" && say "gate v2 (pid $gate) RESUMED (freeze_decision batteries done after ${t}s)"
