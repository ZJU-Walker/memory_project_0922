#!/bin/bash
# stop the waiting seqE chain on hgx-1 (its tests moved to the H200 seqG); the train-split battery is untouched.
pkill -u "$USER" -f "h100_ab_seqE.sh" && echo "seqE stopped $(date +%H:%M)" || echo "seqE not running"
