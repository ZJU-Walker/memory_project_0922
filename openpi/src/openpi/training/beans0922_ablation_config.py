"""Automatic template-slot SNAP and sensory ablations: per-row A250 -> B3000.

A writes label sentences; B writes predicted sentence contents, with no label ramp.
B loads all of its own row\'s A parameters and starts a fresh optimizer. The training
writer uses argmax under teacher forcing (as B9); deployment decodes autoregressively.
Both stages retain label-supervised sentence-history prefill. Auxiliary banks replay
past observations without gradients before a window, and persist across rollout ticks.
The main rows share KI base/10000, seed, v4e sampling/losses and bank decay.
The separate snap_mlp3 control keeps the losses/decay but uses a three-hidden-layer
sentence bank, mean-probability 0.9 write gate, and 25/25/50 sampling with pad 50.
The separate snap_mlp3_a9align row restores the historical A9/B9 layer-8 reader and
training recipe, except RTC15 and fresh KI initialization, with A500 -> B3000.
Only its B stage batches the frozen image tower outside the sequence scan; A stays unchanged.
The paired snap_token_mlp3_a9align row changes only the writer representation to
whitened contextual token keys/standardized token values; no pointer or read-back.
The requested hardware recipes use batch 16 on H200 and batch 12 on H100, no accumulation.
Slot addresses and their count come from the training vocabulary, never task names.
"""

import dataclasses
import os
from pathlib import Path

from openpi.shared import nnx_utils
from openpi.training import beans0922_config as _b
from openpi.training import config as cfg
from openpi.training import weight_loaders
from openpi.models.sentence_slots import template_representatives

PROJECT = "beans0922_ablation"  # W&B project of every ablation row
AB_STEPS = int(os.environ.get("OPENPI_BEANS_AB_STEPS", "3000"))
AB_BATCH = int(os.environ.get("OPENPI_BEANS_AB_BATCH", "16"))
AB_ACCUM = int(os.environ.get("OPENPI_BEANS_AB_ACCUM", "1"))
AB_FSDP = int(os.environ.get("OPENPI_BEANS_AB_FSDP", "4"))
AB_WORKERS = int(os.environ.get("OPENPI_BEANS_AB_WORKERS", "8"))
AB_LABEL_WRITE_STEPS = 0
A_STEPS = int(os.environ.get("OPENPI_BEANS_AB_A_STEPS", "250"))
A9ALIGN_A_STEPS = int(os.environ.get("OPENPI_BEANS_AB_A_STEPS", "500"))
PREFILL_STEPS = int(os.environ.get("OPENPI_BEANS_AB_PREFILL_STEPS", "320"))
RECIPE = "template_slot_ab_v1"
AB_SAVE_EVERY = 250  # rolling checkpoints for resume (the 2 newest are kept)
AB_KEEP_EVERY = int(os.environ.get("OPENPI_BEANS_AB_KEEP", "1000"))  # permanent: 1000 / 2000 / 3000 (~27 GB each); 500 if disk allows
VIS_SLOTS = 8
SNAP_OVERRIDES = dict(
    _b.V4E_ONSET_MODEL,
    memory_template_read=True, memory_v5_slot_keys=True, memory_v5_whiten_values=True,
    memory_v6_token_writes=False, memory_v6_pointer_read=False, memory_v6_whiten_keys=False,
    memory_v0920_prev_readback=False, memory_v0920_query_context=False,
    memory_v5_own_commit_label_content=False,
    memory_v7_write_debounce_steps=1, memory_v7_write_retract_steps=0,
)
SNAP_BANK_OVERRIDES = _b.V4_BANK  # v4: alpha_step 0.001 on both banks (the sensory bank copies memory_semantic, so it follows)


def checkpoint_root() -> Path:
    override = os.environ.get("OPENPI_BEANS_AB_CHECKPOINT_ROOT")
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError("OPENPI_BEANS_AB_CHECKPOINT_ROOT must be an absolute path")
        return path
    return _b._root() / _b.CHECKPOINTS_REL  # noqa: SLF001


def _recipe(existing: dict, name: str, *, steps: int, wandb: bool, batch: int, oracle: bool = False) -> cfg.TrainConfig:
    """snap's memory config under the 4-card ablation recipe (steps, batch, FSDP, checkpoint cadence, W&B project)."""
    references = existing["pi05_yam_mem_v5_beansB9"].model.memory_v5_reference_tokens
    count = len(template_representatives(references))
    overrides = dict(SNAP_OVERRIDES, memory_v5_read_queries=count, memory_v5_oracle_writes=oracle)
    base = _b.memory_config(existing, name, steps=steps, wandb=wandb, batch=batch, model_overrides=overrides,
                            bank_overrides=dict(SNAP_BANK_OVERRIDES, slot_count=count), data_overrides=_b.V4E_ONSET_DATA)
    return dataclasses.replace(
        base, model=dataclasses.replace(base.model, memory=dataclasses.replace(base.model.memory, slot_count=0)),
        fsdp_devices=AB_FSDP, num_workers=AB_WORKERS, project_name=PROJECT,
        gradient_accumulation_steps=AB_ACCUM,
        checkpoint_base_dir=str(checkpoint_root()),
        save_interval=AB_SAVE_EVERY, keep_period=AB_KEEP_EVERY, label_write_schedule_steps=AB_LABEL_WRITE_STEPS,
    )


def snap_config(existing: dict, name: str = "pi05_yam_beans0922_ab_snap", *, steps: int = AB_STEPS, wandb: bool = True,
                batch: int = AB_BATCH) -> cfg.TrainConfig:
    """Control row: template-slot SNAP under the shared ablation recipe."""
    return _recipe(existing, name, steps=steps, wandb=wandb, batch=batch)


def snap_mlp3_config(existing: dict, name: str = "pi05_yam_beans0922_ab_snap_mlp3", *, steps: int = AB_STEPS,
                     wandb: bool = True, batch: int = AB_BATCH) -> cfg.TrainConfig:
    """SNAP-MLP3: change only bank depth, confidence aggregation/gate and window sampling.

    Three 1024-wide hidden layers; the inner delta rule updates only the final
    1024x2048 matrix. Hidden-layer parameters still learn in the outer optimizer.
    RTC 15, direct template reads at the input, v4e losses and decay 0.999 stay as SNAP.
    """
    base = snap_config(existing, name, steps=steps, wandb=wandb, batch=batch)
    return dataclasses.replace(
        base,
        model=dataclasses.replace(
            base.model,
            memory_semantic=dataclasses.replace(base.model.memory_semantic, hidden_dims=(1024, 1024, 1024)),
            memory_v5_write_conf=0.9,
            memory_v5_write_conf_min=False,
        ),
        data=dataclasses.replace(
            base.data,
            base_config=dataclasses.replace(
                base.data.base_config,
                memory_slice_prob=0.5, memory_critical_prob=0.5, memory_critical_start_pad=50,
            ),
        ),
    )


def sensory_config(existing: dict, name: str, *, image: bool = True, state: bool = False, rule: str = "delta",
                   steps: int = AB_STEPS, wandb: bool = True, batch: int = AB_BATCH, slots: int = VIS_SLOTS) -> cfg.TrainConfig:
    """snap + a sensory bank (Pi0Config.memory_vis_bank) read as `slots` fixed-query tokens at the input. `image` = the
    pooled front-camera slots, `state` = the state slot, `rule` = the commit rule of the bank ("delta" or "additive").
    The bank is `model.memory` reconfigured as a copy of the sentence bank's MemoryConfig (linear, decay 0.999, blank start)
    with that commit rule."""
    base = _recipe(existing, name, steps=steps, wandb=wandb, batch=batch)
    model = dataclasses.replace(
        base.model,
        memory=dataclasses.replace(base.model.memory_semantic, slot_count=0, commit_rule=rule),
        memory_vis_bank=True,
        memory_vis_slots=slots,
        memory_vis_image_write=image,
        memory_vis_state_slot=state,
        memory_vis_prefill_steps=PREFILL_STEPS,
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
SENTENCE_ROWS = ("snap", "snap_mlp3")
DIRECT_READ_ROWS = (*SENTENCE_ROWS, *ROWS)
A9ALIGN_ROW = "snap_mlp3_a9align"
TOKEN_A9ALIGN_ROW = "snap_token_mlp3_a9align"
A9ALIGN_ROWS = (A9ALIGN_ROW, TOKEN_A9ALIGN_ROW)
ALL_ROWS = (*DIRECT_READ_ROWS, *A9ALIGN_ROWS)


def stage_a_steps(row: str) -> int:
    return A9ALIGN_A_STEPS if row in A9ALIGN_ROWS else A_STEPS


def default_run_name(row: str) -> str:
    return "token_mlp3_a9align" if row == TOKEN_A9ALIGN_ROW else f"slot_{row}"


def stage_a_params(row: str) -> str:
    return os.environ.get("OPENPI_BEANS_AB_A_PARAMS") or str(
        checkpoint_root() / f"pi05_yam_beans0922_ab_{row}_A" / f"{default_run_name(row)}_A" / str(stage_a_steps(row)) / "params"
    )


def a9align_config(existing: dict, *, stage: str, smoke: bool = False, row: str = A9ALIGN_ROW) -> cfg.TrainConfig:
    """Old A9/B9 with RTC15 and B-only batched vision; KI initialization, A500 -> B3000.

    Clone the historical stage directly: this restores the 8 conditioned reads at
    layer 8, 3x1024 output-delta bank, decay .99, state mask .5, pad75, and original
    losses / stage-specific learning rates. No direct-template/input read path.
    A9's old B6a warm start is unavailable; all memory leaves start fresh from KI.
    B loads every parameter of this experiment's own A500, with a fresh optimizer.
    User 2026-09-24: leave A's execution path untouched; B computes the frozen image
    tower outside the rematerialized tick scan. No parameter or reader-layout change.
    """
    if row not in A9ALIGN_ROWS or stage not in ("A", "B"):
        raise ValueError(f"Invalid aligned row/stage: {row}/{stage}")
    source = existing[f"pi05_yam_mem_v5_beans{stage}9"]
    name = f"pi05_yam_beans0922_ab_{row}" + ("_A" if stage == "A" else "") + ("_smoke" if smoke else "")
    model = dataclasses.replace(source.model, simulated_delay=15, memory_v0920_vision_outside_scan=stage == "B")
    if row == TOKEN_A9ALIGN_ROW:
        # A whole sentence still passes one change/confidence gate. Its valid tokens
        # then commit sequentially, with decay once per tick (not once per token).
        model = dataclasses.replace(
            model, memory_v5_slot_keys=False, memory_v5_whiten_values=False,
            memory_v6_token_writes=True, memory_v6_whiten_keys=True,
        )
    steps = 2 if smoke else (A9ALIGN_A_STEPS if stage == "A" else AB_STEPS)
    loader = weight_loaders.AuditedPartialCheckpointWeightLoader(
        _b.base_params_path(), matched_allowlist=(cfg._V7_NON_MEMORY_LEAF,),  # noqa: SLF001
        fresh_init_allowlist=(cfg._V7_MEMORY_LEAF,), source_cast_dtype="float32",  # noqa: SLF001
    ) if stage == "A" else weight_loaders.AuditedPartialCheckpointWeightLoader(
        stage_a_params(row), matched_allowlist=(r".+",), fresh_init_allowlist=(),
        ignored_source_allowlist=(), source_cast_dtype="float32",
    )
    return dataclasses.replace(
        source, name=name, model=model,
        data=_b._with_beans_data(source.data), weight_loader=loader,  # noqa: SLF001
        checkpoint_base_dir=str(checkpoint_root()),
        assets_base_dir=str(_b._root() / "beans/assets"),  # noqa: SLF001
        # Match this repository's existing zero-based checkpoint naming (Step 0 included).
        num_train_steps=steps + 1, save_interval=AB_SAVE_EVERY,
        keep_period=250 if stage == "A" else AB_KEEP_EVERY, checkpoint_max_to_keep=2,
        batch_size=AB_BATCH, gradient_accumulation_steps=AB_ACCUM,
        fsdp_devices=AB_FSDP, num_workers=AB_WORKERS, seed=42,
        project_name=PROJECT, wandb_enabled=not smoke, log_interval=10, log_diagnostics=False,
        label_write_schedule_steps=0,
    )


def row_config(existing: dict, row: str, *, stage: str = "B", smoke: bool = False) -> cfg.TrainConfig:
    if row in A9ALIGN_ROWS:
        return a9align_config(existing, stage=stage, smoke=smoke, row=row)
    name = f"pi05_yam_beans0922_ab_{row}" + ("_A" if stage == "A" else "") + ("_smoke" if smoke else "")
    kwargs = dict(steps=2 if smoke else (A_STEPS if stage == "A" else AB_STEPS), wandb=not smoke, batch=AB_BATCH)
    if row == "snap":
        base = snap_config(existing, name, **kwargs)
    elif row == "snap_mlp3":
        base = snap_mlp3_config(existing, name, **kwargs)
    else:
        image, state, rule = ROWS[row]
        base = sensory_config(existing, name, image=image, state=state, rule=rule, **kwargs)
    loader = base.weight_loader if stage == "A" else weight_loaders.AuditedPartialCheckpointWeightLoader(
        stage_a_params(row), matched_allowlist=(r".+",), fresh_init_allowlist=(),
        ignored_source_allowlist=(), source_cast_dtype="float32",
    )
    return dataclasses.replace(
        base, model=dataclasses.replace(base.model, memory_v5_oracle_writes=stage == "A"), weight_loader=loader,
        keep_period=A_STEPS if stage == "A" else AB_KEEP_EVERY, seed=42,
    )


def get_configs(existing: dict) -> list:
    return [row_config(existing, row, stage=stage, smoke=smoke)
            for row in ALL_ROWS for stage in ("A", "B") for smoke in (False, True)]
