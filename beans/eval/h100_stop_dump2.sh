#!/bin/bash
# stop the window-dump rerun that started without waiting for seqH (its srun step and python included)
pkill -u "$USER" -f "h100_battery_train_dump2.sh"; pkill -u "$USER" -f "flip_beans0922_v4c_2000_traindump"
sleep 2; pgrep -u "$USER" -af "traindump|dump2" | cut -c1-80; echo "dump2 stopped $(date +%H:%M)"
