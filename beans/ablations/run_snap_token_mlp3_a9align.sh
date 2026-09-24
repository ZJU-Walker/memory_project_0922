#!/usr/bin/env bash
# A9-aligned token-writer control: 3x1024, 8 conditioned reads at layer 8.
# No pointer/read-back. KI base -> own A500 -> B3000, batch 12 on four H100s.
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=${1:-train} CFG=pi05_yam_beans0922_ab_snap_token_mlp3_a9align exec bash "$D/run_stages.sh"
