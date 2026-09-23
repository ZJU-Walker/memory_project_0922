#!/bin/bash
# h100_restart_chain.sh STEP : stop any running h100_eval_chain (videos/battery) on this node and start the chain for STEP.
# A script file on purpose: an inline ssh command that names the chain AND launches it kills its own shell (pgrep self-match).
set -u
STEP=${1:?step}; cd /iris/u/kewalk/memory_project_beans0922 || exit 2
for pat in "h100_eval_chai[n].sh" "run_heldout_video[s].sh" "run_battery_in_jo[b].sh" "v5_heldout_vide[o].py" "v5_count_flip_eva[l].py"; do
  for p in $(pgrep -u "$USER" -f "$pat"); do [ "$p" != "$$" ] && kill -TERM "$p" 2>/dev/null; done
done
sleep 4
for pat in "v5_heldout_vide[o].py" "v5_count_flip_eva[l].py"; do for p in $(pgrep -u "$USER" -f "$pat"); do kill -KILL "$p" 2>/dev/null; done; done
echo "remnants: $(pgrep -u "$USER" -f "v5_heldout_vide[o].py|v5_count_flip_eva[l].py|h100_eval_chai[n].sh" | grep -v "^$$\$" | wc -l)"
ck=beans/checkpoints/pi05_yam_beans0922_v4c/beans0922_v4c/$STEP
[ -f "$ck/_CHECKPOINT_METADATA" ] && [ -d "$ck/params" ] || { echo "checkpoint $STEP not complete"; exit 1; }
setsid nohup bash beans/eval/h100_eval_chain.sh "$STEP" > "beans/eval/h100_v4c_${STEP}.launch.log" 2>&1 < /dev/null &
sleep 2; echo "launched chain for $STEP: $(pgrep -u "$USER" -af "h100_eval_chai[n].sh" | cut -c1-70)"
