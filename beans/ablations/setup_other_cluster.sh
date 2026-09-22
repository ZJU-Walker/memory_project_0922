#!/usr/bin/env bash
# Set up the beans0922 tree on another machine in one go (clone, venv, data + checkpoint, tokenizer caches).
#   bash setup_other_cluster.sh [<dest dir>]          # default ./memory_project_beans0922
#   LOCAL_DISK=/local/ssd bash setup_other_cluster.sh  # + keep the dataset and the loader's arrow cache on the node's own disk
#                                                      #   (do this when the clone is on a network filesystem: ~3-4x faster loading)
# Needs: git, uv (https://docs.astral.sh/uv/ -- `curl -LsSf https://astral.sh/uv/install.sh | sh`), python 3.11 (uv fetches
# it), ~70 GB disk, internet to GitHub, huggingface.co and storage.googleapis.com (anonymous). Then, on a 4-GPU node:
#   GPUS=0,1,2,3 bash <dest>/beans/ablations/run_vis8.sh smoke     # 2-update launch check
#   GPUS=0,1,2,3 bash <dest>/beans/ablations/run_vis8.sh           # the run (detach it with nohup / tmux / your scheduler)
set -euo pipefail
DEST="${1:-$PWD/memory_project_beans0922}"
REPO="${REPO:-https://github.com/ZJU-Walker/memory_project_0922.git}"
command -v uv >/dev/null 2>&1 || { echo "uv not found; install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 2; }
if [ -d "$DEST/.git" ]; then echo "repo present: $DEST"; else git clone "$REPO" "$DEST"; fi
cd "$DEST/openpi"
# --no-install-package rerun-sdk: lerobot's visualizer, unused here; its wheel needs glibc >= 2.31 and RHEL-8-class nodes
# (glibc 2.28, e.g. Purdue Anvil) have none -- everything else in the lock installs there
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-package rerun-sdk
LOCAL_DISK="${LOCAL_DISK:-}" bash "$DEST/beans/ablations/00_download.sh"
echo "setup complete: $DEST"
