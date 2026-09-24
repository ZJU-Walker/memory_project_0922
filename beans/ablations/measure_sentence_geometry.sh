#!/usr/bin/env bash
# Read-only model diagnostics on CPU; creates only the explicitly named report directory.
set -euo pipefail
ROOT="${MEMORY_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "$ROOT"
export MEMORY_PROJECT_ROOT="$ROOT" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/v35/cache/huggingface" HF_DATASETS_CACHE="$ROOT/v35/cache/huggingface/datasets"
export HF_LEROBOT_HOME="$ROOT/data/lerobot" OPENPI_DATA_HOME="$ROOT/v35/cache/openpi"
export OPENPI_JAX_CACHE_DIR="$ROOT/v35/cache/jax" UV_CACHE_DIR="$ROOT/v35/cache/uv"
exec "${OPENPI_PYTHON:-$ROOT/openpi/.venv/bin/python}" "$ROOT/beans/ablations/sentence_geometry.py" "$@"
