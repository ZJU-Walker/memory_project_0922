#!/bin/bash
# episode 29 at v4c/1750 on H100 card 0 next to the running chain: label notes (oracle) and empty bank (self + blank).
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
export JOB=17489557 GPU=0 GRES=2 CPUS=6 MEMFRAC=0.3
MODES=oracle EPISODES=29 bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v4c beans0922_v4c 1750
MODES=self EPISODES=29 EXTRA="--intervention blank" bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v4c beans0922_v4c 1750
