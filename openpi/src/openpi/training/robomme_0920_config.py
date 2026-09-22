"""RoboMME PickXtimes 0920_v0 (robomme/docs/0920_v0_plan.md; worktree memory_project_0920, branch v0920).

One self-write run from the ORIGINAL pi0.5 base weights (user 2026-09-20): no stage A/B, the sentence bank is read with
8 fixed learned queries at the INPUT (visible to every block; no conditioner, no pointer bonus, no "Last:" prompt slot),
written every tick with no rule, and the prefix carries a short visual history (4 past front-camera frames, 0.2 s apart,
pooled to 64 tokens each, per-slot time embedding) in place of the blank third camera. Tick = 20 control steps (1 s),
40-tick windows (40 s), action horizon 40. A label-write probability 1 -> 0 over the first 3k of 10k updates keeps the
read path alive while the model's own notes are still noise (TrainConfig.label_write_schedule_steps).

Everything else (official sentences, token-level whitened keys, linear delta-rule bank with 0.99 decay, sampling mix,
KI-style token CE with onset weight 3, SigLIP + embedder frozen) is the v3 recipe (robomme_config._mem_variant).

v1 (09-21 14:32 "remove history image so still 2 images"; 15:08 "make action horizon prediction to 30 instead of 40 ...
bigger batch ... make sure this v1 start from plain base pi05"): the same run WITHOUT the past frames and WITHOUT the blank
third camera slot -- the prefix is front + wrist only (2 x 256 image tokens + 80 prompt + 8 read = 600 tokens per tick vs
856 in v0), action horizon 30 (1.5 s; 20 executed at serving), batch from the H200 probe (OPENPI_0920_V1_BATCH), the same
pi05_base loader with fresh memory leaves, same steps / schedule / ramp. Configs pi05_robomme_0920_v1 / _smoke / _probe and
_binfill_v1 / _smoke. Motivation: the 09-21 history-mask diagnostic (robomme/replay/val50_0920_v0_2000_nohist) -- the arm
ignores the past frames while the notes advance on a clock with them (notes-right 37 % with history vs 65 % without);
v3B picked the cube 9/10 with two cameras; one frame + state gives the up/down phase 97.8 % (pick_updown_from_one_frame.py).
"""

import dataclasses
import os

from openpi.shared import project_paths
from openpi.training import config as cfg
from openpi.training import optimizer, weight_loaders
from openpi.training import robomme_config as _rc

# the original pi05 base (openpi-assets), hard-linked into THIS tree from memory_project_robomme/v35/cache (the training
# identity check refuses paths outside the project root; a symlinked openpi-assets breaks the gs:// loader)
PI05_BASE_PARAMS = os.environ.get(
    "OPENPI_PI05_BASE_PARAMS",
    str(project_paths.memory_project_root() / "v35/cache/openpi/openpi-assets/checkpoints/pi05_base/params"),
)
V0_TICK_FRAMES = 20  # memory tick = 1 s at the 20 Hz benchmark control
V0_SEQ_STEPS = 40  # 40 ticks = 800 frames = 40 s windows (v3: 160 x 5)
V0_BLOCK_STEPS = 20  # TBPTT fence (20 s)
V0_HISTORY_FRAMES = 4  # past front frames per tick ...
V0_HISTORY_STRIDE = 4  # ... 4 frames (0.2 s) apart: t-16, t-12, t-8, t-4
# 09-20 23:45 (user: "yes lets do batchsize 32 with 5k steps"): the batch-32 probe ran 20 s/update vs 11 s at batch 8 on the
# 4 x H200 (2.2x the windows per hour), so batch 32 with half the updates: same ~28 h wall time, twice the data seen.
V0_STEPS = 5_000
V0_LABEL_WRITE_STEPS = 1_500  # label-write probability 1 -> 0 over the first 30 % of updates
V0_BATCH = 32

V0_MODEL = dict(
    action_horizon=40,
    max_token_len=80,  # "Task: <goal>, State: <8 digits>;\n" measures <= 64 tokens over all 100 goals; no slot
    memory_seq_steps=V0_SEQ_STEPS,
    memory_block_steps=V0_BLOCK_STEPS,
    prompt_slot_len=0,
    prompt_slot_dropout=0.0,
    # read: 8 fixed queries at the input, no visual columns, no pointer, no conditioning shifts
    memory_v7_no_visual_block=True,
    memory_v4_visual_injection=False,
    memory_v0920_input_read=True,
    memory_v0920_input_rms=None,  # = RMS of the reference-sentence word embeddings (measured from the embedder)
    memory_v0920_history_frames=V0_HISTORY_FRAMES,
    memory_v0920_history_pool=2,  # 16x16 -> 8x8 = 64 tokens per past frame
    memory_v0920_history_dropout=0.2,
    memory_v0920_vision_outside_scan=os.environ.get("OPENPI_0920_VISION_HOIST", "1") == "1",  # H100 A/B 09-21: 12.5 vs 15.1 s/step
    remat_policy=os.environ.get("OPENPI_0920_REMAT", "nothing_saveable"),  # per-block remat inside the per-tick checkpoint; A/B knob (09-21)
    memory_v6_pointer_read=False,
    memory_v5_query_prev_sentence=False,
    memory_v5_query_standardize=False,
    memory_mask_zero_tokens=True,
    memory_blind_tokens=True,
    # write: every valid tick, no confidence / debounce / grammar / vocabulary / retraction rule
    memory_v5_oracle_writes=False,
    memory_v7_write_every_step=True,
    memory_v5_write_conf=0.0,
    memory_v7_write_debounce_steps=1,
    memory_v7_first_write_debounce_steps=None,
    memory_v7_write_retract_steps=0,
    memory_v7_write_vocab_only=False,
    memory_v7_write_grammar=(),
    memory_v5_prev_is_committed=True,
    memory_v5_own_commit_label_content=False,
    memory_v5_write_delay_steps=0,
    memory_v5_prefill_history=True,  # slices start with the label history in the bank (as a rollout would have its own)
    memory_v5_prefill_max=16,
    # losses / regularisation as v3B
    memory_state_mask_prob=0.5,
    memory_v7_hard_token_ce_weight=1.0,
    memory_v7_onset_ce_weight=3.0,
)
V0_DATA = dict(
    memory_stride_frames=V0_TICK_FRAMES,
    memory_min_slice_steps=10,
    memory_sequence_buckets=(10, 20, 30, 40),
    memory_slice_prob=0.4,
    memory_critical_prob=0.4,
    memory_critical_start_pad=5 * V0_TICK_FRAMES,  # 5 ticks before a label transition (v3: 75 frames = 5 ticks of 15)
    memory_start_history_power=1.0,
    memory_image_history_frames=V0_HISTORY_FRAMES,
    memory_image_history_stride=V0_HISTORY_STRIDE,
    memory_image_history_key="image",
)
V1_HISTORY_FRAMES = 0  # v1: no past frames, no blank slot (the two real cameras only)
V1_ACTION_HORIZON = 30  # v1: 1.5 s plans (user 09-21 15:08), 20 steps executed at serving as before
V1_BATCH = int(os.environ.get("OPENPI_0920_V1_BATCH", "32"))  # set from the H200 probe (robomme/logs/probe_0920_v1_h200.sh)


def _history_overrides(history: int) -> tuple[dict, dict]:
    """(model, data) overrides for `history` past front frames per tick. V0_HISTORY_FRAMES (4) = the V0 dicts as they are;
    0 (v1) = no history keys and no blank third slot: the model sees the two real cameras only (RobommeInputs drops the masked
    right-wrist slot), the loader fetches one frame per tick, the history dropout has nothing to drop."""
    if history == V0_HISTORY_FRAMES:
        return {}, {}
    if history != 0:
        raise ValueError(f"0920 history frames must be {V0_HISTORY_FRAMES} (v0) or 0 (v1), got {history}.")
    return (dict(memory_v0920_history_frames=0, memory_v0920_history_dropout=0.0, memory_v0920_drop_blank_camera=True),
            dict(memory_image_history_frames=0))


def _v0(root, spec, existing, name: str, *, steps: int, save_every: int, keep: int, max_keep: int, wandb: bool,
        history: int = V0_HISTORY_FRAMES, horizon: int = V0_MODEL["action_horizon"], batch: int = V0_BATCH):
    m_over, d_over = _history_overrides(history)
    base = _rc._mem_variant(  # noqa: SLF001
        root, spec, existing, name, oracle_writes=False, loader_path=PI05_BASE_PARAMS,
        matched=(_rc.NON_MEMORY_LEAF,), fresh=(_rc.MEMORY_LEAF,), steps=steps, save_every=save_every, keep=keep,
        peak_lr=2.5e-5, max_keep=max_keep, tag="official", model_overrides=dict(V0_MODEL, action_horizon=horizon, **m_over),
        data_overrides=dict(V0_DATA, **d_over),
    )
    return dataclasses.replace(
        base,
        project_name="robomme_0920",  # a new W&B project (user 09-20: the old log was too messy)
        batch_size=batch,  # v0: eight windows per GPU on the 4 x H200 (probe: 20 s/update, fits with headroom); v1: V1_BATCH
        fsdp_devices=4,
        num_workers=24,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=V0_STEPS, decay_lr=2.5e-6),
        label_write_schedule_steps=V0_LABEL_WRITE_STEPS,
        wandb_enabled=wandb,
        log_interval=10,
        log_diagnostics=False,
    )


def get_configs(existing: dict) -> list:
    root = project_paths.memory_project_root()
    configs = []
    spec = _rc._load_spec(root, "official")  # noqa: SLF001
    if spec is not None:
        configs += [
            _v0(root, spec, existing, "pi05_robomme_0920_v0", steps=V0_STEPS + 1, save_every=500, keep=1_000, max_keep=2, wandb=True),
            # smoke-only: 3 updates, no W&B, same recipe (traces the data path, the loader audit and the compiled step)
            _v0(root, spec, existing, "pi05_robomme_0920_v0_smoke", steps=3, save_every=1_000, keep=1_000, max_keep=1, wandb=False),
            # timing probe (09-21): 31 updates, ONE window length (every window padded to 40 ticks -> one compiled shape), no W&B;
            # step time = updates 10..30. Used on the 2 x H100 pair for A/B speed tests (batch/fsdp from the launcher).
            _probe(root, spec, existing),
            # v1 (09-21): two real cameras only, horizon 30, batch from the probe; same steps / schedule / ramp / base weights
            _v0(root, spec, existing, "pi05_robomme_0920_v1", steps=V0_STEPS + 1, save_every=500, keep=1_000, max_keep=2, wandb=True,
                **V1_KW),
            _v0(root, spec, existing, "pi05_robomme_0920_v1_smoke", steps=3, save_every=1_000, keep=1_000, max_keep=1, wandb=False,
                **V1_KW),
            _probe(root, spec, existing, name="pi05_robomme_0920_v1_probe", **V1_KW),
        ]
    spec_b = _rc._load_spec(root, "official", task=BINFILL_DATASET)  # noqa: SLF001
    if spec_b is not None:
        configs += binfill_configs(root, spec_b, existing)
    return configs


V1_KW = dict(history=V1_HISTORY_FRAMES, horizon=V1_ACTION_HORIZON, batch=V1_BATCH)


def _probe(root, spec, existing, name: str = "pi05_robomme_0920_v0_probe", **kw):
    cfg = _v0(root, spec, existing, name, steps=31, save_every=10_000, keep=10_000, max_keep=1, wandb=False, **kw)
    return dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, memory_sequence_buckets=(40,)))
    )


# ---- BinFill (09-21) ---------------------------------------------------------------------------------------------------
# User: "start training the binfill task using exactly same setup? make full use of that 2 h100" and "adjust the window
# length and our sample rate, if this task is shorter we can include more full eps". Same recipe (V0_MODEL / V0_DATA, one
# self-write run from the pi05 base, label-write ramp, history frames, every-tick writes); only the task data and the
# window geometry change. Released BinFill episodes are 266-1044 frames (median 620; PickXtimes 266-1025, median 543): the
# PickXtimes window (40 ticks x 20 frames = 800 frames) would cut 23 of 100 BinFill episodes, so the tick becomes 25 frames
# (1.25 s) and the window 42 ticks = 1050 frames >= the longest episode -> every episode fits one window at ~5 % more
# sequence than the PickXtimes run (tick 20 would need 53 ticks, +33 %). Shortest label segment is 32 frames (> one tick).
# Data = the released 100 demos + 100 generated with the benchmark recorder (cluster_robomme/generate_robomme_demos.py),
# converted together by cluster_robomme/prepare_robomme_h5_to_lerobot.py (dataset / metadata / assets name BINFILL_DATASET).
BINFILL_DATASET = os.environ.get("OPENPI_0920_BINFILL_DATASET", "BinFill200")  # BinFill100 = released only (probes)
BINFILL_TICK_FRAMES = 25
BINFILL_SEQ_STEPS = 42
BINFILL_BLOCK_STEPS = 21  # TBPTT fence at half the window (PickXtimes: 20 of 40)
BINFILL_BUCKETS = (12, 22, 32, 42)
BINFILL_BATCH = 8  # 2 x H100 (80 GB): 4 windows per card at fsdp 2 (H200 run: 8 per card at 141 GB); OPENPI_0920_BINFILL_BATCH overrides
BINFILL_WORKERS = 16


def binfill_sets(sentences) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Label-derived sampling sets (the PickXtimes counterpart is robomme_config._count_sentences): evidence = the
    sentences that open a new object cycle (they carry the running count: "pick up the second red cube"), decision =
    those plus the stop ("press the button"). Only sampling / state-mask bookkeeping; nothing in the model."""
    picks = tuple(s for s in sentences if s.startswith("pick up the"))
    stop = tuple(s for s in sentences if s.startswith("press the button"))
    return picks, picks + stop


def _v0_task(root, spec, existing, name: str, *, task: str, steps: int, save_every: int, keep: int, max_keep: int, wandb: bool,
             tick: int, seq_steps: int, block_steps: int, buckets: tuple[int, ...], batch: int, fsdp: int, workers: int,
             history: int = V0_HISTORY_FRAMES, horizon: int = V0_MODEL["action_horizon"]):
    """The `_v0` recipe on another RoboMME task/dataset: same model and data knobs, the task's own paths, sentence sets
    and window geometry (tick, window, TBPTT block, buckets, critical pad) and the launch-hardware batch/fsdp/workers;
    `history` as in `_v0` (4 = v0 past frames, 0 = v1 two cameras only)."""
    evidence, decision = binfill_sets(spec["sentences"])
    meta = root / project_paths.ROBOMME_METADATA_DIR / task
    m_over, d_over = _history_overrides(history)
    model_overrides = dict(
        V0_MODEL, memory_seq_steps=seq_steps, memory_block_steps=block_steps, action_horizon=horizon,
        memory_v5_prefill_max=max(16, int(spec["max_segments_per_episode"]) + 1),  # boba rule: sentences per episode + 1
        **m_over,
    )
    data_overrides = dict(
        V0_DATA, memory_stride_frames=tick, memory_sequence_buckets=buckets, memory_critical_start_pad=5 * tick,
        repo_id=f"robomme/{task}", evidence_subtasks=evidence, memory_required_subtasks=decision,
        memory_episode_manifest_path=str(meta / "manifest.json"),
        memory_v5_subtask_labels_path=str(meta / "subtasks_official.json"),
        **d_over,
    )
    base = _rc._mem_variant(  # noqa: SLF001
        root, spec, existing, name, oracle_writes=False, loader_path=PI05_BASE_PARAMS,
        matched=(_rc.NON_MEMORY_LEAF,), fresh=(_rc.MEMORY_LEAF,), steps=steps, save_every=save_every, keep=keep,
        peak_lr=2.5e-5, max_keep=max_keep, tag="official", model_overrides=model_overrides, data_overrides=data_overrides,
    )
    data = dataclasses.replace(
        base.data, repo_id=f"robomme/{task}",
        assets=cfg.AssetsConfig(assets_dir=str(root / project_paths.ROBOMME_ASSETS_ROOT), asset_id=task),
    )
    return dataclasses.replace(
        base,
        data=data,
        project_name="robomme_0920",
        batch_size=batch,
        fsdp_devices=fsdp,
        num_workers=workers,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=V0_STEPS, decay_lr=2.5e-6),
        label_write_schedule_steps=V0_LABEL_WRITE_STEPS,
        wandb_enabled=wandb,
        log_interval=10,
        log_diagnostics=False,
    )


def binfill_configs(root, spec, existing) -> list:
    common = dict(task=BINFILL_DATASET, tick=BINFILL_TICK_FRAMES, seq_steps=BINFILL_SEQ_STEPS, block_steps=BINFILL_BLOCK_STEPS,
                  buckets=BINFILL_BUCKETS, batch=int(os.environ.get("OPENPI_0920_BINFILL_BATCH", BINFILL_BATCH)), fsdp=2,
                  workers=BINFILL_WORKERS)
    probe = _v0_task(root, spec, existing, "pi05_robomme_0920_binfill_v0_probe", steps=31, save_every=10_000, keep=10_000,
                     max_keep=1, wandb=False, **common)
    probe = dataclasses.replace(
        probe, data=dataclasses.replace(probe.data, base_config=dataclasses.replace(
            probe.data.base_config, memory_sequence_buckets=(BINFILL_SEQ_STEPS,))))
    return [
        _v0_task(root, spec, existing, "pi05_robomme_0920_binfill_v0", steps=V0_STEPS + 1, save_every=500, keep=1_000, max_keep=2,
                 wandb=True, **common),
        _v0_task(root, spec, existing, "pi05_robomme_0920_binfill_v0_smoke", steps=3, save_every=1_000, keep=1_000, max_keep=1,
                 wandb=False, **common),
        probe,  # one compiled shape (every window padded to 42 ticks): batch / step-time probe on the H100 pair
        # v1 (09-21): two real cameras only, horizon 30 (see the module docstring); batch stays the H100 value
        _v0_task(root, spec, existing, "pi05_robomme_0920_binfill_v1", steps=V0_STEPS + 1, save_every=500, keep=1_000, max_keep=2,
                 wandb=True, history=V1_HISTORY_FRAMES, horizon=V1_ACTION_HORIZON, **common),
        _v0_task(root, spec, existing, "pi05_robomme_0920_binfill_v1_smoke", steps=3, save_every=1_000, keep=1_000, max_keep=1,
                 wandb=False, history=V1_HISTORY_FRAMES, horizon=V1_ACTION_HORIZON, **common),
    ]
