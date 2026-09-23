#!/bin/bash
# kill any running onset A/B process, then start the drop set (script file on purpose: no pgrep self-match).
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
for pat in "h100_onset_ab_dro[p].sh" "h100_onset_a[b].sh" "v5_onset_a[b].py"; do
  for p in $(pgrep -u "$USER" -f "$pat"); do [ "$p" != "$$" ] && kill -TERM "$p" 2>/dev/null; done
done
sleep 3; for p in $(pgrep -u "$USER" -f "v5_onset_a[b].py"); do kill -KILL "$p" 2>/dev/null; done; sleep 1
echo "remnants: $(pgrep -u "$USER" -f "v5_onset_a[b].py|h100_onset_ab_dro[p].sh" | grep -v "^$$\$" | wc -l)"
setsid nohup bash beans/eval/h100_onset_ab_drop.sh > beans/eval/h100_onset_ab_drop.launch.log 2>&1 < /dev/null &
sleep 1; echo "started: $(pgrep -u "$USER" -af "h100_onset_ab_dro[p].sh" | cut -c1-60)"
