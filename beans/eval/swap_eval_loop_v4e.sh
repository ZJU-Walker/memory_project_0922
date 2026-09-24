#!/bin/bash
# replace eval_loop_v4e.sh (A/B only) by eval_loop_v4e2.sh (A/B + two own-note rollouts) on hgx-1.
# Pattern anchored on the directory: a bare "eval_loop_v4e.sh" also matched THIS script's own name (swap_eval_loop_v4e.sh)
# and killed the remote shell (17:10, rc=255 twice).
pkill -u "$USER" -f "eval/eval_loop_v4e[.]sh"; sleep 2
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
if ! pgrep -u "$USER" -f "eval/eval_loop_v4e2[.]sh" >/dev/null; then
  setsid nohup bash /iris/u/kewalk/memory_project_beans0922/beans/eval/eval_loop_v4e2.sh > /iris/u/kewalk/memory_project_beans0922/beans/eval/eval_loop_v4e2.out 2>&1 < /dev/null &
  sleep 2
fi
echo "loops now:"; pgrep -u "$USER" -af "eval/eval_loop_v4e" | cut -c1-90
