#!/bin/bash
#SBATCH --job-name=robomme_eval_a40
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=/iris/u/kewalk/memory_project_robomme/robomme/logs/slurm/%x-%j.out
#SBATCH --error=/iris/u/kewalk/memory_project_robomme/robomme/logs/slurm/%x-%j.err
# RoboMME eval allocation (user 09-15 17:37 "use a40 to eval, you can submit slurm yourself using iris"): holds one
# A40 for the checkpoint batteries of the PickXtimes self-write runs. The batch step only keeps the allocation alive
# (the user's 1 GB train_hs.py keep-alive, same as on every job of theirs); evals run as --overlap steps from the node:
#   JOB=<this job> GPU=0 cluster_robomme/run_mem_evals.sh <config> <exp> <step>
set -u
echo "==== Job $SLURM_JOB_ID on $SLURMD_NODENAME $(date) ===="
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
cd /iris/u/kewalk/openpi_trossen || exit 2
export HOME=/iris/u/kewalk
exec /iris/u/kewalk/openpi_trossen/.venv/bin/python3 cluster_scripts/train_hs.py
