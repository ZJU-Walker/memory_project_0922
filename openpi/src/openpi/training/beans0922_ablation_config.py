"""beans0922 ablations (2026-09-22) -- variants of the LED bean-scoop memory policy ("snap", beans0922_config.memory_config)
for the ablation table, every one trained with the SAME 4-card recipe so the rows differ in one thing only.

snap = the beans0922 **v4** structure since 2026-09-23 00:35 (user, relayed by the base session: every ablation row adopts the
same read/write changes; v3 was adopted 09-22 15:30 and is superseded -- its question shift collapsed the 8 questions into one):
v1 + beans0922_config.V4_TOKEN_EXACT + V4_BANK = change-only own writes gated on the LOWEST token probability >= 0.8
(memory_v7_write_every_step False, memory_v5_write_conf_min, memory_v5_write_conf 0.8), the pointer bonus on the sentence read
(memory_v6_pointer_read, context queries, beta 10), the last committed note read back through the bank as 48 extra input tokens
after the 8 question tokens (memory_v0920_prev_readback), a slower decay for BOTH banks (alpha_step 0.001 = 0.999 per tick) and
the error-driven sentence-token weight 5 (memory_v7_hard_token_ce_weight). No question context (v3's shift is off). The sensory
bank is unaffected by all of it except the decay it shares: it writes every valid tick (training scan and serving transition
gate it on tick validity, not on the sentence write decision), its 8 fixed read queries have no pointer bonus, and its tokens
are appended after the read-back tokens (memory block = 8 questions | 48 read-back | 8 sensory = 64 tokens). Rows started
before 09-23 00:35 (the 09-22 Anvil vis8s / vis8 runs, the 09-22 13:24 Stanford vis8 run) used the v3 or v1 snap and are not
comparable with the v4 rows.

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
from the launcher (beans/ablations/train_ablation.sh, default 16 with an OOM fallback ladder), checkpoints every 250 (the two
newest kept for resume) with 1000 / 2000 / 3000 permanent (OPENPI_BEANS_AB_KEEP=500 for a finer eval grid; ~27 GB each). Everything else (window, tick, labels, sampling, losses) is beans0922_config.memory_config.

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
AB_SAVE_EVERY = 250  # rolling checkpoints for resume (the 2 newest are kept)
AB_KEEP_EVERY = int(os.environ.get("OPENPI_BEANS_AB_KEEP", "1000"))  # permanent: 1000 / 2000 / 3000 (~27 GB each); 500 if disk allows
VIS_SLOTS = 8
SNAP_OVERRIDES = _b.V4_TOKEN_EXACT  # the snap revision every row is built on (v4; see the module docstring)
SNAP_BANK_OVERRIDES = _b.V4_BANK  # v4: alpha_step 0.001 on both banks (the sensory bank copies memory_semantic, so it follows)


def _recipe(existing: dict, name: str, *, steps: int, wandb: bool, batch: int) -> cfg.TrainConfig:
    """snap's memory config under the 4-card ablation recipe (steps, batch, FSDP, checkpoint cadence, W&B project)."""
    base = _b.memory_config(existing, name, steps=steps, wandb=wandb, batch=batch, model_overrides=SNAP_OVERRIDES,
                            bank_overrides=SNAP_BANK_OVERRIDES)
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
