#!/bin/bash
# 16:31 user: problem located, no more tests. Stop every eval chain and its srun/python on this node (keep-alives untouched).
for pat in h200_ab_seqG.sh h200_ab_seqF.sh h100_battery_train_dump3.sh h100_ab_seqH.sh v5_onset_ab.py v5_count_flip_eval.py; do pkill -u "$USER" -f "$pat"; done
sleep 4; pkill -9 -u "$USER" -f "v5_onset_ab.py"; pkill -9 -u "$USER" -f "v5_count_flip_eval.py"; sleep 2
echo "$(hostname) left: $(pgrep -u "$USER" -af 'seq[FGH]|dump[3]|v5_onset_a[b]|v5_count_fli[p]' | wc -l)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
