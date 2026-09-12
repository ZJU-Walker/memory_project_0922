"""Smoke test of the v7 phase-context data path on the real boba dataset: build the torch dataset of a
config, pull a few consecutive frames of one episode and print the decoded prompt prefix. CPU only."""
import sys
import numpy as np
import torch  # noqa: F401  (import order: torch before tensorflow, see data_loader_test.py)
import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
from openpi.models import tokenizer as _tokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "pi05_yam_boba0911_ctx_both"
cfg = _config.get_config(name)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
ds = _data_loader.transform_dataset(ds, data_config)
pg = _tokenizer.FASTSubtaskTokenizer(cfg.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
print(f"{name}: {len(ds)} frames; max_token_len {cfg.model.max_token_len}")
for idx in [0, 10, 40, 700, 705, 1200]:
    item = ds[idx]
    toks = np.asarray(item["tokenized_prompt"]); m = np.asarray(item["tokenized_prompt_mask"]); ar = np.asarray(item["token_ar_mask"])
    n = int(m.sum()); n_prefix = int((m & (ar == 0)).sum())
    text = pg.decode(toks[:n_prefix].tolist())
    print(f"[{idx}] tokens {n}/{len(toks)} (prefix {n_prefix}) :: {text[:60]} ... {text[-95:]!r}")
    assert "state_history" not in item and "prev_subtask" not in item
    if n >= len(toks):
        raise SystemExit(f"FULL BUFFER at idx {idx}: {n} >= {len(toks)} (truncation!)")
print("ok")
