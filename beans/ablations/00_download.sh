#!/usr/bin/env bash
# Run ONCE per machine (after `uv sync` in openpi/): fetch the LED bean-scoop dataset, its norm stats and the beans0922
# knowledge-insulation base checkpoint from the Hugging Face Hub into this tree's DEFAULT locations (so no env override is
# needed afterwards), then pre-fetch the tokenizers into the tree's own caches (train.py pins HF_HOME / OPENPI_DATA_HOME
# there). Public repos, no token needed. ~60 GB. Safe to re-run: existing pieces are skipped.
#   bash beans/ablations/00_download.sh
# LOCAL_DISK=<dir on the node's own disk> (optional, recommended when the clone sits on a network filesystem): the dataset is
# downloaded there and the arrow cache the loader builds on first start lives there too; the tree gets symlinks. Measured on
# our cluster: dataset + arrow cache on NFS 2 s/update, on local disk 1.8 updates/s (the loader memory-maps the arrow cache).
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT/openpi"
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"; HF="$(dirname "$PY")/huggingface-cli"
[ -x "$PY" ] || { echo "no venv at $PY -- run: cd $ROOT/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen"; exit 2; }
DATASET_REPO=kewalk123/yam_bean_scoop_0905_v5
BASE_REPO=kewalk123/beans0922_pi05_base_10k
DS_DIR="$ROOT/v5/data/lerobot/yam/bean_scoop_0905_v5"                       # beans0922_config.DATASET_ROOT_REL
ASSETS="$ROOT/v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5"  # beans0922_config.ASSETS_DIR_REL
BASE_ROOT="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base"  # <step>/params under it; 10000 = beans0922_config.base_params_path()
for link in v5 data v6; do  # this cluster's trees use symlinks into sibling checkouts; a dangling one must not shadow the download
  if [ -L "$ROOT/$link" ] && [ ! -e "$ROOT/$link" ]; then echo "removing dangling symlink $ROOT/$link"; rm "$ROOT/$link"; fi
done
if [ -n "${LOCAL_DISK:-}" ]; then
  # the tree's path guard sanctions symlinks under local/ (node-local mirrors) and under v35/cache (caches) only
  mkdir -p "$LOCAL_DISK/beans0922/hf_datasets" "$LOCAL_DISK/beans0922/bean_scoop_0905_v5" "$ROOT/local" "$ROOT/v35/cache/huggingface"
  DS_DIR="$ROOT/local/bean_scoop_0905_v5"; [ -e "$DS_DIR" ] || ln -s "$LOCAL_DISK/beans0922/bean_scoop_0905_v5" "$DS_DIR"
  echo "dataset -> $DS_DIR -> $LOCAL_DISK/beans0922/bean_scoop_0905_v5 (train_ablation.sh picks local/ up automatically)"
  ARROW="$ROOT/v35/cache/huggingface/datasets"  # train.py pins HF_DATASETS_CACHE to this path; a symlink keeps the cache local
  if [ -d "$ARROW" ] && [ ! -L "$ARROW" ]; then rmdir "$ARROW" 2>/dev/null || mv "$ARROW" "$ARROW.nfs_$(date +%s)"; fi
  [ -e "$ARROW" ] || ln -s "$LOCAL_DISK/beans0922/hf_datasets" "$ARROW"; echo "arrow cache -> $LOCAL_DISK/beans0922/hf_datasets"
fi
if [ -f "$DS_DIR/meta/info.json" ] && [ -n "$(ls "$DS_DIR/data" 2>/dev/null)" ]; then echo "dataset present: $DS_DIR"; else
  echo "downloading $DATASET_REPO -> $DS_DIR (89 episodes, ~56 GB)"; mkdir -p "$DS_DIR"
  "$HF" download "$DATASET_REPO" --repo-type dataset --local-dir "$DS_DIR"
fi
mkdir -p "$ASSETS"; cp -n "$DS_DIR/openpi_assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json" "$ASSETS/" 2>/dev/null || true
[ -f "$ASSETS/norm_stats.json" ] || { echo "norm stats missing under $DS_DIR/openpi_assets"; exit 1; }
# the base checkpoint: the Hub holds params/ of ONE step plus a STEP file (the final 10000 since 2026-09-22 08:30 PDT; 5000
# was published while the base run was still going); it lands at <step>/params and train_ablation.sh warm-starts from the
# largest step present. A machine that downloaded step 5000 earlier re-runs this script once to add 10000.
STEP="$("$HF" download "$BASE_REPO" STEP --local-dir "$ROOT/beans/checkpoints/.hf_base_meta" >/dev/null 2>&1 && tr -d '[:space:]' < "$ROOT/beans/checkpoints/.hf_base_meta/STEP")"
[ -n "$STEP" ] || { echo "the base repo has no STEP / params yet (the base checkpoint upload is missing)"; exit 1; }
BASE_DIR="$BASE_ROOT/$STEP"
if [ -d "$BASE_DIR/params" ] && [ -n "$(ls "$BASE_DIR/params" 2>/dev/null)" ]; then echo "base checkpoint present: $BASE_DIR/params"; else
  echo "downloading $BASE_REPO (step $STEP) -> $BASE_DIR"; mkdir -p "$BASE_DIR"
  "$HF" download "$BASE_REPO" --local-dir "$BASE_DIR" --exclude "STEP" "README.md"
fi
[ -d "$BASE_DIR/params" ] || { echo "download of $BASE_REPO failed"; exit 1; }
echo "pre-fetching tokenizers into the tree's caches"
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets" OPENPI_DATA_HOME="$ROOT/v35/cache/openpi" OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax" UV_CACHE_DIR="$ROOT/v35/cache/uv" HF_LEROBOT_HOME="$ROOT/data/lerobot"
MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu "$PY" - <<'PYEOF'
from openpi.shared import project_paths
project_paths.configure_v35_runtime_environment()
from openpi.models import tokenizer
tokenizer.FASTSubtaskTokenizer(80)  # PaliGemma tokenizer (GCS, anonymous) + physical-intelligence/fast (Hub)
print("tokenizers cached")
PYEOF
echo "done: dataset $DS_DIR | norm stats $ASSETS | base step $STEP at $BASE_DIR/params"
