#!/usr/bin/env bash
# A9-aligned slot SNAP + sensory state, independent KI -> A500 -> B3000.
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_state8_mlp3_a9align exec bash "$D/run_stages.sh"
