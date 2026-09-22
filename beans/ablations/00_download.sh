#!/usr/bin/env bash
# Run ONCE per machine (after `uv sync` in openpi/): fetch the LED bean-scoop dataset, its norm stats and the beans0922
# knowledge-insulation base checkpoint from the Hugging Face Hub into this tree's DEFAULT locations (so no env override is
# needed afterwards), then pre-fetch the tokenizers into the tree's own caches (train.py pins HF_HOME / OPENPI_DATA_HOME
# there). Public repos, no token needed. ~60 GB. Safe to re-run: existing pieces are skipped.
#   bash beans/ablations/00_download.sh
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"; cd "$ROOT/openpi"
PY="${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}"; HF="$(dirname "$PY")/huggingface-cli"
[ -x "$PY" ] || { echo "no venv at $PY -- run: cd $ROOT/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen"; exit 2; }
DATASET_REPO=kewalk123/yam_bean_scoop_0905_v5
BASE_REPO=kewalk123/beans0922_pi05_base_10k
DS_DIR="$ROOT/v5/data/lerobot/yam/bean_scoop_0905_v5"                       # beans0922_config.DATASET_ROOT_REL
ASSETS="$ROOT/v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5"  # beans0922_config.ASSETS_DIR_REL
BASE_DIR="$ROOT/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000"  # beans0922_config.base_params_path()
for link in v5 data v6; do  # this cluster's trees use symlinks into sibling checkouts; a dangling one must not shadow the download
  if [ -L "$ROOT/$link" ] && [ ! -e "$ROOT/$link" ]; then echo "removing dangling symlink $ROOT/$link"; rm "$ROOT/$link"; fi
done
if [ -f "$DS_DIR/meta/info.json" ] && [ -n "$(ls "$DS_DIR/data" 2>/dev/null)" ]; then echo "dataset present: $DS_DIR"; else
  echo "downloading $DATASET_REPO -> $DS_DIR (89 episodes, ~56 GB)"; mkdir -p "$DS_DIR"
  "$HF" download "$DATASET_REPO" --repo-type dataset --local-dir "$DS_DIR"
fi
mkdir -p "$ASSETS"; cp -n "$DS_DIR/openpi_assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json" "$ASSETS/" 2>/dev/null || true
[ -f "$ASSETS/norm_stats.json" ] || { echo "norm stats missing under $DS_DIR/openpi_assets"; exit 1; }
if [ -d "$BASE_DIR/params" ]; then echo "base checkpoint present: $BASE_DIR/params"; else
  echo "downloading $BASE_REPO -> $BASE_DIR"; mkdir -p "$BASE_DIR"
  "$HF" download "$BASE_REPO" --local-dir "$BASE_DIR"
fi
[ -d "$BASE_DIR/params" ] || { echo "the base repo has no params/ yet (pushed once the base run reaches 10k)"; exit 1; }
echo "pre-fetching tokenizers into the tree's caches"
MEMORY_PROJECT_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu "$PY" - <<'PYEOF'
from openpi.shared import project_paths
project_paths.configure_v35_runtime_environment()
from openpi.models import tokenizer
tokenizer.FASTSubtaskTokenizer(80)  # PaliGemma tokenizer (GCS, anonymous) + physical-intelligence/fast (Hub)
print("tokenizers cached")
PYEOF
echo "done: dataset $DS_DIR | norm stats $ASSETS | base $BASE_DIR/params"
