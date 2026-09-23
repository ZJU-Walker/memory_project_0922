#!/bin/bash
# label-note (oracle) rollouts of v4c/1750 on TRAINING x=1 episodes 3 and 8 (+ x=2 episode 2), H100 card 0 of job 17489557.
cd /iris/u/kewalk/memory_project_beans0922 || exit 2
export JOB=17489557 GPU=0 GRES=2 CPUS=8 MEMFRAC=0.6
MODES=oracle EPISODES="3 8 2" bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v4c beans0922_v4c 1750
