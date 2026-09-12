#!/bin/bash
# Token-length audit for the boba base config (scripts/v33_audit_token_lengths.py): iterates every frame of the
# transformed dataset with an oversized budget and reports the true maxima of the context and causal segments;
# the sum (+ slack) is what pi05_yam_boba0911_base.max_token_len must be (provisional 320). Needs the norm stats.
# CPU only; run as an srun --overlap step on a node; the step writes its own log.
set -euo pipefail
LOG=/iris/u/kewalk/memory_project_v6/v6/logs/audit_tokens_boba_v1.log
exec >> "$LOG" 2>&1
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
cd /iris/u/kewalk/memory_project_v6/openpi
find src/openpi scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[$(date)] host $(hostname) token audit for pi05_yam_boba0911_base"
.venv/bin/python scripts/v33_audit_token_lengths.py --config-name pi05_yam_boba0911_base --num-workers 16 --batch-size 64
echo "[$(date)] audit finished rc=$?"
touch /iris/u/kewalk/memory_project_v6/v6/logs/audit_tokens_boba_v1.done
