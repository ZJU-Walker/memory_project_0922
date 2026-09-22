#!/usr/bin/env bash
# stop every beans0922 process on this node (watcher, chain, launcher, train.py); a script file so pkill never matches its caller
pkill -u "$USER" -f 'beans/logs/start_after_convert[.]sh'; pkill -u "$USER" -f 'beans/logs/chain_beans0922[.]sh'
pkill -u "$USER" -f 'beans/logs/train_beans0922[.]sh'; pkill -u "$USER" -f 'train[.]py pi05_yam_beans0922'; sleep 3
echo "left: $(pgrep -u "$USER" -af 'beans0922|start_after_conver[t]' | grep -v stop_all | wc -l)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
