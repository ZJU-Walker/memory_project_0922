#!/bin/bash
# 16:19: an orphaned onset A/B (from the seqE chain) shares card 0 and the same log file with seqH's run.
# Kill every A/B on this node, then relaunch seqH and the window dump behind it.
echo "orphan args:"; ps -u "$USER" -o pid,ppid,args | grep "v5_onset_a[b]" | sed 's/.*--sidecar [^ ]* //' | cut -c1-120
ps -o pid,ppid,args -p 1430483 2>/dev/null | tail -1 | cut -c1-120
pkill -u "$USER" -f "h100_ab_seqE.sh"; pkill -u "$USER" -f "h100_ab_seqH.sh"; pkill -u "$USER" -f "h100_battery_train_dump3.sh"
pkill -u "$USER" -f "h100_onset_ab.sh"; pkill -u "$USER" -f "v5_onset_ab.py"; sleep 5
pkill -9 -u "$USER" -f "v5_onset_ab.py" 2>/dev/null; sleep 2
echo "left:"; pgrep -u "$USER" -af "onset_ab|seq[EH]|dump[3]" | cut -c1-80
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
setsid nohup bash /iris/u/kewalk/memory_project_beans0922/beans/eval/h100_ab_seqH.sh > /iris/u/kewalk/memory_project_beans0922/beans/eval/h100_ab_seqH.out 2>&1 < /dev/null &
sleep 3
setsid nohup bash /iris/u/kewalk/memory_project_beans0922/beans/eval/h100_battery_train_dump3.sh > /iris/u/kewalk/memory_project_beans0922/beans/eval/h100_battery_train_dump3.out 2>&1 < /dev/null &
sleep 2; echo "relaunched:"; pgrep -u "$USER" -af "seq[H]|dump[3]" | cut -c1-80; echo "reset done $(date +%H:%M)"
