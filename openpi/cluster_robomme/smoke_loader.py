"""CPU smoke test of the RoboMME data path (no model): build the torch dataset of a config, pull items and check the
contract -- 8-D state/actions padded to 32, the next-action offset (chunk k starts at frame k*stride+1), the zero
right-wrist mask, the v2 sentences, the prefill fields and the context token budget. Usage:
  .venv/bin/python cluster_robomme/smoke_loader.py pi05_robomme_mem_PickXtimes_B [pi05_robomme_PickXtimes_base_ki]
"""
import sys

import numpy as np
import torch  # noqa: F401  (import order: torch before tensorflow)
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.models import tokenizer as _tokenizer


def check(name: str) -> None:
    cfg = _config.get_config(name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    ds = _data_loader.transform_dataset(ds, data_config)
    pg = _tokenizer.FASTSubtaskTokenizer(cfg.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    memory = getattr(cfg.model, "predict_with_memory", False)
    print(f"\n== {name}: {len(ds)} items, memory={memory}, max_token_len={cfg.model.max_token_len}, "
          f"offset={data_config.action_target_offset_frames}, stride={data_config.memory_stride_frames}")
    for idx in ([0, 1, 250, 4000] if memory else [0, 114, 115, 528, 529, 20000]):
        item = ds[idx]
        st, ac = np.asarray(item["state"]), np.asarray(item["actions"])
        toks, m, ar = (np.asarray(item[k]) for k in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask"))
        # memory items carry [T, L] token rows: budget per step; non-memory items one [L] row
        n = int(m.sum(-1).max()); n_prefix = int((m & (ar == 0)).sum(-1).max())
        text = pg.decode(toks[:n_prefix].tolist()) if not memory else pg.decode(toks[0][: int((m[0] & (ar[0] == 0)).sum())].tolist())
        mask = item["image_mask"]
        rw = np.asarray(mask["right_wrist_0_rgb"])
        print(f"[{idx}] state {st.shape} actions {ac.shape} img {np.asarray(item['image']['base_0_rgb']).shape} "
              f"right_wrist_mask_any={bool(rw.any())} tokens {n}/{toks.shape[-1]} prefix {n_prefix}")
        print(f"      context: {text[:140]!r}")
        assert not rw.any(), "right wrist must be masked out"
        assert st.shape[-1] == 32 and ac.shape[-1] == 32
        assert np.all(st[..., 8:] == 0) and np.all(ac[..., 8:] == 0), "pad dims must be zero"
        if memory:
            steps = st.shape[0]
            sm = np.asarray(item["seq_step_mask"])
            pf = np.asarray(item["memory_v5_prefill_mask"]).any(-1).sum() if "memory_v5_prefill_mask" in item else "n/a"
            dec = np.asarray(item["seq_decision_mask"]).sum() if "seq_decision_mask" in item else "n/a"
            # the per-step sentence targets are the ar=1 causal rows: decode the first step's target
            causal = toks[0][int((m[0] & (ar[0] == 0)).sum()):int(m[0].sum())]
            print(f"      steps {steps} valid {int(sm.sum())} prefill_rows={pf} decision_steps={dec} "
                  f"keys={sorted(k for k in item if k.startswith(('seq_', 'memory_v5')))}")
            print(f"      step0 causal: {pg.decode([int(t) for t in causal[:20]])[:100]!r}")
            if n >= toks.shape[-1]:
                raise SystemExit(f"FULL CONTEXT BUFFER at idx {idx}: {n} >= {toks.shape[-1]}")
        else:
            print(f"      subtask={item.get('subtask')!r}")
            if n >= toks.shape[-1]:
                raise SystemExit(f"FULL BUFFER at idx {idx}: {n} >= {toks.shape[-1]} (truncation!)")
    print("ok", name)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["pi05_robomme_mem_PickXtimes_B", "pi05_robomme_PickXtimes_base_ki"]:
        check(name)
