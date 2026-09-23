#!/bin/bash
# stop every offline eval process of ours on this node (video chains, single rollouts, batteries); training is untouched.
for pat in "h100_eval_chai[n].sh" "h100_ep29_[a-z]*.sh" "run_heldout_video[s].sh" "run_battery_in_jo[b].sh" "v5_heldout_vide[o].py" "v5_count_flip_eva[l].py"; do
  for p in $(pgrep -u "$USER" -f "$pat"); do [ "$p" != "$$" ] && kill -TERM "$p" 2>/dev/null; done
done
sleep 4
for pat in "v5_heldout_vide[o].py" "v5_count_flip_eva[l].py"; do for p in $(pgrep -u "$USER" -f "$pat"); do kill -KILL "$p" 2>/dev/null; done; done
echo "remaining eval processes: $(pgrep -u "$USER" -f "v5_heldout_vide[o].py|v5_count_flip_eva[l].py|h100_eval_chai[n].sh|run_heldout_video[s].sh" | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
