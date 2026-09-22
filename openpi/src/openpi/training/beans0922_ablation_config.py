"""beans0922 ablations (2026-09-22) -- variants of the LED bean-scoop memory policy ("snap", beans0922_config.memory_config)
for the ablation table, every one trained with the SAME 4-card recipe so the rows differ in one thing only.

  pi05_yam_beans0922_ab_snap   snap itself under the ablation recipe (the control row): 8 sentence-bank read tokens.
  pi05_yam_beans0922_ab_vis8   ablation (1) "snap + visual memory": a second fast-weight bank fed by the front camera
                               (Pi0Config.memory_vis_bank, 8 slots) and read with 8 fixed queries at the input next to
                               the 8 sentence tokens; the bank is `Pi05Config.memory` reconfigured as the SAME linear
                               delta-rule bank as the sentence bank (decay 0.99 per tick, rate 1, blank start).
  pi05_yam_beans0922_ab_vis8s  vision + state: the same bank with one more write slot per tick from the 14-D state
                               (memory_vis_state_slot); delta rule (a presence memory: repeats add nothing).
  pi05_yam_beans0922_ab_vis8s_add   vision + state with the ADDITIVE commit rule (memory.commit_rule = "additive",
                               Hebbian / outer product: repeats accumulate, a tally memory), same decay, same read.
  pi05_yam_beans0922_ab_state8      state slot only (memory_vis_image_write False), delta rule, 8 read tokens.
  pi05_yam_beans0922_ab_state8_add  state slot only, additive rule.
  <name>_smoke                 2 updates, no W&B, for the launch check.

Ablation recipe (user 2026-09-22 04:39: "make full use ... batch size larger ... target 3k steps and same first 500 label
descend"): warm start from the beans0922 knowledge-insulation base (base/10000, OPENPI_BEANS_BASE_PARAMS) with fresh memory
leaves, 3000 updates, label-write probability 1 -> 0 over the first 500 updates, lr 2.5e-5, FSDP over the 4 cards, batch
from the launcher (beans/ablations/train_ablation.sh, default 16 with an OOM fallback ladder), checkpoints every 250 (every
500 kept). Everything else (window, tick, labels, sampling, losses) is beans0922_config.memory_config.

Env knobs: OPENPI_BEANS_AB_STEPS / _BATCH / _FSDP / _WORKERS (defaults 3000 / 16 / 4 / 16) plus the beans0922 dataset /
checkpoint overrides (OPENPI_BEANS_DATASET_ROOT, OPENPI_BEANS_ASSETS_DIR, OPENPI_BEANS_BASE_PARAMS).
"""

import dataclasses
import os

from openpi.shared import nnx_utils
from openpi.training import beans0922_config as _b
from openpi.training import config as cfg

PROJECT = "beans0922_ablation"  # W&B project of every ablation row
AB_STEPS = int(os.environ.get("OPENPI_BEANS_AB_STEPS", "3000"))
AB_BATCH = int(os.environ.get("OPENPI_BEANS_AB_BATCH", "16"))
AB_FSDP = int(os.environ.get("OPENPI_BEANS_AB_FSDP", "4"))
AB_WORKERS = int(os.environ.get("OPENPI_BEANS_AB_WORKERS", "16"))
AB_LABEL_WRITE_STEPS = _b.MEM_LABEL_WRITE_STEPS  # 500, as snap
AB_SAVE_EVERY = 250
AB_KEEP_EVERY = 500
VIS_SLOTS = 8


def _recipe(existing: dict, name: str, *, steps: int, wandb: bool, batch: int) -> cfg.TrainConfig:
    """snap's memory config under the 4-card ablation recipe (steps, batch, FSDP, checkpoint cadence, W&B project)."""
    base = _b.memory_config(existing, name, steps=steps, wandb=wandb, batch=batch)
    return dataclasses.replace(
        base, fsdp_devices=AB_FSDP, num_workers=AB_WORKERS, project_name=PROJECT,
        save_interval=AB_SAVE_EVERY, keep_period=AB_KEEP_EVERY, label_write_schedule_steps=AB_LABEL_WRITE_STEPS,
    )


def snap_config(existing: dict, name: str = "pi05_yam_beans0922_ab_snap", *, steps: int = AB_STEPS, wandb: bool = True,
                batch: int = AB_BATCH) -> cfg.TrainConfig:
    """The control row: snap, unchanged, under the ablation recipe."""
    return _recipe(existing, name, steps=steps, wandb=wandb, batch=batch)


def sensory_config(existing: dict, name: str, *, image: bool = True, state: bool = False, rule: str = "delta",
                   steps: int = AB_STEPS, wandb: bool = True, batch: int = AB_BATCH, slots: int = VIS_SLOTS) -> cfg.TrainConfig:
    """snap + a sensory bank (Pi0Config.memory_vis_bank) read as `slots` fixed-query tokens at the input. `image` = the
    pooled front-camera slots, `state` = the state slot, `rule` = the commit rule of the bank ("delta" or "additive").
    The bank is `model.memory` reconfigured as a copy of the sentence bank's MemoryConfig (linear, decay 0.99, blank start)
    with that commit rule."""
    base = _recipe(existing, name, steps=steps, wandb=wandb, batch=batch)
    model = dataclasses.replace(
        base.model,
        memory=dataclasses.replace(base.model.memory_semantic, commit_rule=rule),
        memory_vis_bank=True,
        memory_vis_slots=slots,
        memory_vis_image_write=image,
        memory_vis_state_slot=state,
    )
    # snap freezes its sentence gate at tanh(w) = 0.5 (memory_sem_inject_w in the freeze filter); the sensory gate gets the
    # same treatment so the two banks inject at the same fixed scale and the rows stay controlled
    pattern = base.freeze_filter.pattern.pattern
    if "memory_sem_inject_w" not in pattern:
        raise ValueError("expected snap's freeze filter to freeze memory_sem_inject_w")
    freeze = nnx_utils.PathRegex(pattern.replace("memory_sem_inject_w", "memory_sem_inject_w|memory_vis_inject_w", 1), sep=base.freeze_filter.sep)
    return dataclasses.replace(base, model=model, freeze_filter=freeze)


def vis8_config(existing: dict, name: str = "pi05_yam_beans0922_ab_vis8", *, steps: int = AB_STEPS, wandb: bool = True,
                batch: int = AB_BATCH, slots: int = VIS_SLOTS) -> cfg.TrainConfig:
    """Ablation (1): snap + an `slots`-slot visual bank (image slots only, delta rule)."""
    return sensory_config(existing, name, image=True, state=False, rule="delta", steps=steps, wandb=wandb, batch=batch, slots=slots)


ROWS = {  # name suffix -> (image slots, state slot, commit rule); every row = snap + this bank under the same recipe
    "vis8": (True, False, "delta"),
    "vis8s": (True, True, "delta"),
    "vis8s_add": (True, True, "additive"),
    "state8": (False, True, "delta"),
    "state8_add": (False, True, "additive"),
}


def get_configs(existing: dict) -> list:
    configs = [snap_config(existing), snap_config(existing, "pi05_yam_beans0922_ab_snap_smoke", steps=2, wandb=False)]
    for suffix, (image, state, rule) in ROWS.items():
        name = f"pi05_yam_beans0922_ab_{suffix}"
        configs.append(sensory_config(existing, name, image=image, state=state, rule=rule))
        configs.append(sensory_config(existing, f"{name}_smoke", image=image, state=state, rule=rule, steps=2, wandb=False))
    return configs
