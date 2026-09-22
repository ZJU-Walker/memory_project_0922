"""RoboMME PickXtimes on the v7 boba memory recipe (cluster_robomme/README.md, worktree memory_project_robomme).

Only data adapters and configs live here; the model implementation is the v7 one. Registered by
openpi.training.config at import time (`get_configs`), after every v7 config exists, and only when the prepared
label descriptor `robomme/metadata/PickXtimes/prepared_v2.json` is present (its SHA256 pins are re-checked).

Recipe (user 2026-09-13 18:17 / 18:22):
  * copy of the boba memory repo (memory_project_v7 @ ab09cac), adapted to RoboMME: NO RTC (`simulated_delay=None`,
    the benchmark rollout is synchronous), memory write opportunities every 10 frames, analytic label-history
    prefill ON (short 40-step windows instead of whole episodes; the bank before a window is initialised from the
    label history, exactly as boba/task1), window length chosen for the task (PickXtimes episodes are 27-103 steps
    at stride 10; a pick-and-place cycle is <= 27 steps, so 40 steps = 400 frames spans a cycle and a half).
  * two-phase labels v2 (place carries the count of its pick; cluster_robomme/build_pickxtimes_v2_labels.py).
  * no stage A: like boba2B, the memory stage starts from the task's NON-memory pi05+KI base with own write timing
    + label content. The user's "PickXtimes base 9999" does not exist -- the legacy copy only has a PLAIN pi05 base
    (no subtask head, checkpoint 7000, memory_v6_robomme), so `pi05_robomme_PickXtimes_base_ki` first continues
    that checkpoint with knowledge insulation on the v2 sentences (the boba base recipe), then
    `pi05_robomme_mem_PickXtimes_B` is boba2B on top of it. `_B_blind` (09-15) adds the boba5B
    ordinal-row blinding (memory_v7_digit_blind, pi0.py apply_digit_blind) and is the variant to launch. `_B_plain7000` is the literal "skip everything" variant
    (memory stage straight from the plain base; its sentence head starts untrained) and `_A` the oracle-write
    fallback; neither is the plan.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import re

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import robomme_policy, yam_policy
from openpi.shared import project_paths
from openpi.training import config as cfg
from openpi.training import optimizer, weight_loaders

TASK = "PickXtimes"
REPO_ID = "robomme/PickXtimes_v2"
# v3 (2026-09-19): window length in memory ticks (x 5 frames), stage A / stage B update counts (final checkpoint = N).
# Fixed here (not env) so serving rebuilds the same config by name; sized on the 4 x H200 probe (robomme/logs/probe_v3_h200.sh).
V3_SEQ_STEPS = 160  # 04:27 probe: 120-tick windows at batch 8 peak 102.5 of 144 GB -> the headroom goes into 160-tick windows (40 s)
V3_BLOCK_STEPS = 40  # TBPTT fence in ticks (10 s): v2 cut gradients every 25 ticks of 15 frames; longer chains risk the recurrent blow-up
V3_A_STEPS = 400
V3_B_STEPS = 1500
_PICK = re.compile(r"^pick up the \w+ cube for the (\w+) time$")
_PLACE = re.compile(r"^place the \w+ cube onto the target(?: for the (\w+) time)?$")  # official form carries no count
# Target-carry sentences (label version "tgt", 09-17): "pick up the red cube, 2 of 3" / "place the red cube onto the
# target, 2 of 3" -- every phase names its count k and the episode target x (the bean-scoop B9 label design).
_PICK_T = re.compile(r"^pick up the \w+ cube, (\d) of (\d)$")
_PLACE_T = re.compile(r"^place the \w+ cube onto the target, (\d) of (\d)$")
# Label versions (09-15, user 15:00 "this time i still want to use the official provided subtask"): "official" = the
# released simple_subgoal sentences, lowercased (only the pick carries the ordinal; 20 sentences); "v2" = the two-phase
# relabel (the place carries its pick's ordinal; 32 sentences). Same boundaries, same frames; the official configs
# carry the `_off` infix, the v2 names are unchanged.
_LABEL_TAGS = {"official": "_off", "v2": "", "shift1": "_shift1"}  # shift1 = eval-only counterfactual (ordinals +1), never train it
# Ordinal-row blinding (boba5B port): token suffixes ending at the position that PREDICTS the ordinal. PaliGemma ids of
# "cube for the" (pick) and "target for the" (v2 place only); the same rows cluster_robomme/derive_memory_rows.py
# finds from the labels alone (the only tokens the text does not fix but the history does).
_BLIND_PATTERNS = {"official": ((28660, 604, 573),), "v2": ((28660, 604, 573), (4408, 604, 573)), "shift1": ((28660, 604, 573),)}
# Same leaf split as the v7 boba stages: everything the base trained is matched, every memory leaf is fresh.
NON_MEMORY_LEAF = r"(?!.*(?:memory|fact_|query_compressor|query_conditioner|state_null_embedding|probe_head|ladder_)).+"
MEMORY_LEAF = r".*(?:memory|fact_|query_compressor|query_conditioner|state_null_embedding|probe_head|ladder_).*"


@dataclasses.dataclass(frozen=True)
class LeRobotRobommeDataConfig(cfg.LeRobotYamDataConfig):
    """The YAM sequence/label pipeline with two cameras and absolute 8-D Panda state/actions."""

    def create(self, assets_dirs: pathlib.Path, model_config) -> cfg.DataConfig:
        data = super().create(assets_dirs, model_config)
        structure = dict(data.repack_transforms.inputs[0].structure)
        structure.pop("observation/right_wrist_image")  # the benchmark has no right wrist camera
        history = int(data.memory_image_history_frames)  # 0920_v0: past front frames in, blank camera out
        model_history = int(getattr(model_config, "memory_v0920_history_frames", 0))
        if history != model_history:
            raise ValueError(
                f"memory_image_history_frames={history} (data) must equal memory_v0920_history_frames={model_history} (model)."
            )
        # 0920_v0 drops the blank slot to make room for the history tokens; 0920_v1 drops it with no history at all
        drop_blank = history > 0 or bool(getattr(model_config, "memory_v0920_drop_blank_camera", False))
        inputs = [
            robomme_policy.RobommeInputs(model_type=model_config.model_type, history_frames=history, drop_blank_camera=drop_blank)
            if isinstance(t, yam_policy.YamInputs) else t
            for t in data.data_transforms.inputs
            if not isinstance(t, transforms.DeltaActions)  # released actions are absolute joint targets
        ]
        return dataclasses.replace(
            data,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(structure)]),
            data_transforms=transforms.Group(inputs=inputs, outputs=[robomme_policy.RobommeOutputs()]),
        )


def _load_spec(root: pathlib.Path, tag: str, task: str = TASK) -> dict | None:
    """The prepared label descriptor of `task` (robomme/metadata/<task>/prepared_<tag>.json) with its SHA256 pins
    re-checked; `task` names the metadata / assets folder (a dataset name such as BinFill200 for the 0920 BinFill run)."""
    meta = root / project_paths.ROBOMME_METADATA_DIR / task
    spec_path = meta / f"prepared_{tag}.json"
    if not spec_path.is_file():
        return None
    spec = json.loads(spec_path.read_text())
    for name, key in (("manifest.json", "manifest_sha256"), (f"subtasks_{tag}.json", f"subtasks_{tag}_sha256")):
        actual = hashlib.sha256((meta / name).read_bytes()).hexdigest()
        if actual != spec[key]:
            raise ValueError(f"RoboMME {task} {name} was modified after {spec_path.name} was written")
    norm = root / project_paths.ROBOMME_ASSETS_ROOT / task / "norm_stats.json"
    if hashlib.sha256(norm.read_bytes()).hexdigest() != spec["norm_stats_sha256"]:
        raise ValueError(f"RoboMME {task} norm_stats.json was modified after {spec_path.name} was written")
    return spec


def _count_sentences(sentences: tuple[str, ...]) -> tuple[str, ...]:
    """Sentences whose choice needs the count held in the bank: every k >= 2 phase and the final button press."""
    return tuple(
        s for s in sentences
        if ((m := _PICK.match(s) or _PLACE.match(s)) and m.group(1) not in (None, "first")) or s == "press the button to stop"
    )


def _base_data(root: pathlib.Path, spec: dict) -> LeRobotRobommeDataConfig:
    return LeRobotRobommeDataConfig(
        repo_id=REPO_ID,
        base_config=cfg.DataConfig(
            repo_id=REPO_ID,
            prompt_from_episode_meta=True,  # setup/task_goal[0] per episode (meta/episode_prompts.json)
            subtask_from_task=True,  # the v2 sentence is the dataset's per-frame task column
            subtask_lookahead=0,
            action_target_offset_frames=1,  # observation row t supervises actions from row t+1 (recorder order)
            lerobot_dataset_root=str(root / spec["lerobot_dataset"]),
        ),
        assets=cfg.AssetsConfig(assets_dir=str(root / project_paths.ROBOMME_ASSETS_ROOT), asset_id=TASK),
    )


def _base_ki_config(root: pathlib.Path, spec: dict, tag: str) -> cfg.TrainConfig:
    """Non-memory pi05 + knowledge insulation on the v2 sentences (the boba base recipe), continued from the legacy
    plain PickXtimes checkpoint 7000 (already an action policy for this task; its vision tower saw the simulator)."""
    return cfg.TrainConfig(
        name=f"pi05_robomme_{TASK}_base_ki{_LABEL_TAGS[tag]}",
        project_name="robomme_memory",  # W&B project (user 09-15: "enable wandb for training")
        model=pi0_config.Pi0Config(
            pi05=True,
            predict_subtask=True,
            simulated_delay=None,  # RoboMME: synchronous rollout, no RTC
            # context "Task: <goal>, State: <8 values>;" is ~70 tokens (legacy audit: 80 sufficed for the memory
            # path); the non-memory path packs context + sentence (<= 12) + FAST action tokens into one buffer.
            max_token_len=224,
        ),
        data=_base_data(root, spec),
        assets_base_dir=str(root / project_paths.ROBOMME_ASSETS_ROOT),
        checkpoint_base_dir=str(root / project_paths.ROBOMME_CHECKPOINTS_DIR),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "OPENPI_ROBOMME_PLAIN_BASE_PARAMS",
                str(root / project_paths.ROBOMME_CHECKPOINTS_DIR / f"pi05_robomme_base_{TASK}/legacy_plain_r1/7000/params"),
            )
        ),
        batch_size=16,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=5_000,
        save_interval=1_000,
        keep_period=1_000,
        num_workers=8,
        fsdp_devices=2,
        wandb_enabled=False,
        log_interval=10,  # user 09-20: losses every 10 updates, no diagnostic metrics
        log_diagnostics=False,
    )


def _mem_variant(
    root: pathlib.Path, spec: dict, existing: dict, name: str, *, oracle_writes: bool, loader_path: str,
    matched: tuple[str, ...], fresh: tuple[str, ...], steps: int, save_every: int, keep: int, peak_lr: float, max_keep: int = 1,
    model_overrides: dict | None = None, tag: str = "v2", reinit: tuple[str, ...] = (), data_overrides: dict | None = None,
) -> cfg.TrainConfig:
    """The v7 `_v7_boba_mem_variant` recipe (v6.1 A2 template: linear delta-rule sentence bank, token-level causal
    keys whitened over the reference vocabulary, context-query pointer read, analytic history prefill, semantic-only
    freeze) with the RoboMME knobs."""
    base_cfg = existing["pi05_yam_mem_v6_task1A2"]
    sentences = tuple(spec["sentences"])
    tokens = tuple(tuple(int(t) for t in row) for row in spec["reference_tokens"])
    if tag == "tgt":  # target-carry sentences carry digits, not ordinals
        evidence = tuple(s for s in sentences if _PICK_T.match(s) or _PLACE_T.match(s))
        decision = tuple(s for s in sentences if ((m := _PICK_T.match(s) or _PLACE_T.match(s)) and int(m.group(1)) >= 2)
                         or s == "press the button to stop")
    else:
        evidence = tuple(s for s in sentences if _PICK.match(s) or _PLACE.match(s))
        decision = _count_sentences(sentences)
    model_kwargs = dict(
        simulated_delay=None,  # no RTC
        max_token_len=96,  # memory-path context buffer (legacy PickXtimes ran at 80; SwingXtimes needed 96)
        memory_seq_steps=40,  # 400 frames at stride 10 (boba: 60 x 15 frames)
        memory_block_steps=20,  # TBPTT fence (boba: 30 of 60)
        memory_v5_oracle_writes=oracle_writes,
        memory_v5_prev_is_committed=not oracle_writes,
        memory_v5_own_commit_label_content=not oracle_writes,
        memory_v5_prefill_history=True,  # analytic label-history prefill before every window
        memory_v5_prefill_max=int(spec["max_segments_per_episode"]) + 1,  # boba rule: sentences per episode + 1
        memory_v5_reference_tokens=tokens,
        memory_state_mask_prob=0.0,
    )
    model_kwargs.update(model_overrides or {})  # overrides win (e.g. own content, blinding)
    model = dataclasses.replace(base_cfg.model, **model_kwargs)
    data_kwargs = dict(
        repo_id=REPO_ID,
        prompt_from_episode_meta=True,
        subtask_from_task=True,
        subtask_lookahead=0,
        action_target_offset_frames=1,
        memory_stride_frames=10,  # user: memory write every 10 frames
        memory_slice_prob=0.9,
        memory_min_slice_steps=10,
        memory_sequence_buckets=(20, 30, 40),
        evidence_subtasks=evidence,
        memory_required_subtasks=decision,
        memory_critical_prob=0.5,
        memory_critical_start_pad=50,  # 5 steps before a label transition (boba: 75 frames = 5 steps at 15)
        memory_subtask_vocab=sentences,
        memory_waiting_state_dim=None,  # YAM-only anti-leak state masking; 8-D Panda state
        memory_episode_manifest_path=str(root / project_paths.ROBOMME_METADATA_DIR / TASK / "manifest.json"),
        memory_episode_manifest_sha256=spec["manifest_sha256"],
        memory_manifest_split="train",  # all 100 released demonstrations; the benchmark val/test split is the eval
        memory_manifest_split_seed=0,
        memory_v5_subtask_labels_path=str(root / project_paths.ROBOMME_METADATA_DIR / TASK / f"subtasks_{tag}.json"),
        memory_v5_subtask_labels_sha256=spec[f"subtasks_{tag}_sha256"],
        memory_v5_generic_task=True,
        memory_v6_still_decision_boost=1.0,
        memory_v6_still_decision_frames=0,
        memory_v6_tail_sentences=(),
        lerobot_dataset_root=str(root / spec["lerobot_dataset"]),
    )
    data_kwargs.update(data_overrides or {})  # bean sampling for the v1 _lin twins (stride 15, half full starts, ...)
    data = dataclasses.replace(
        _base_data(root, spec),
        base_config=dataclasses.replace(base_cfg.data.base_config, **data_kwargs),
    )
    return dataclasses.replace(
        base_cfg,
        name=name,
        project_name="robomme_memory",  # W&B project (user 09-15: "enable wandb for training")
        model=model,
        data=data,
        assets_base_dir=str(root / project_paths.ROBOMME_ASSETS_ROOT),
        checkpoint_base_dir=str(root / project_paths.ROBOMME_CHECKPOINTS_DIR),
        batch_size=4,  # two 40-step windows per GPU on 2 GPUs (boba2B: batch 4 of 60-step windows)
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=peak_lr, decay_steps=10_000, decay_lr=peak_lr),
        weight_loader=weight_loaders.AuditedPartialCheckpointWeightLoader(
            loader_path,
            matched_allowlist=matched,
            fresh_init_allowlist=fresh,
            reinit_allowlist=reinit,
            ignored_source_allowlist=(),
            source_cast_dtype="float32",
        ),
        num_train_steps=steps,
        save_interval=save_every,
        keep_period=keep,
        checkpoint_max_to_keep=max_keep,
        num_workers=8,
        fsdp_devices=2,
        wandb_enabled=False,
        log_interval=10,  # user 09-20: losses every 10 updates, no diagnostic metrics
        log_diagnostics=False,
    )


def _beans_variant(
    root: pathlib.Path, spec: dict, existing: dict, name: str, *, oracle_writes: bool, loader_path: str,
    matched: tuple[str, ...], fresh: tuple[str, ...], steps: int, save_every: int, keep: int, peak_lr: float,
    max_keep: int = 2, fsdp: int = 1, batch: int = 2, model_overrides: dict | None = None,
) -> cfg.TrainConfig:
    """v1 (09-17, user: "replicate what we have for the blink bean scoop task training, i want to make this task work
    first"): the bean-scoop recipe of memory_project_v5 `pi05_yam_mem_v5_beansA9` (label writes) / `beansB9` (own
    writes) on the PickXtimes target-carry labels. Copied from B9 through its config: Titans MLP sentence bank
    (hidden 3x1024, delta output rule) -- addressed token after token (v6 keys, user 14:08) instead of slot keys --, standardized-attention pooling (4 queries),
    standardized read query conditioned on the previous sentence, label-history prefill (max 16), write delay 0,
    retry-until-committed own writes (B), state masking 0.5, 40-step windows / 25-step TBPTT blocks, stride 5,
    half of the windows from the episode start, 14/27/40-step buckets, critical starts 75 frames before a label
    change, semantic-only freeze, AdamW clip 1.0, memory grad clip 5.0, no EMA, batch 2 per GPU. Not copied: the RTC
    delay 6 (synchronous simulator), max_token_len 80 -> 96 (longer goals), the 14-D YAM waiting-state mask (Panda
    state), the B6a warm start (no memory model exists for this task: the plain 7000 supplies everything but the fresh
    memory leaves, as in v0). None of the v6 (token writes, pointer read) or v7 (debounce, retraction, vocabulary
    gate, hard-word weight) knobs are on."""
    base_cfg = existing["pi05_yam_mem_v5_beansB9"]
    sentences = tuple(spec["sentences"])
    tokens = tuple(tuple(int(t) for t in row) for row in spec["reference_tokens"])
    evidence = tuple(s for s in sentences if _PICK_T.match(s) or _PLACE_T.match(s))
    decision = tuple(
        s for s in sentences
        if ((m := _PICK_T.match(s) or _PLACE_T.match(s)) and int(m.group(1)) >= 2) or s == "press the button to stop"
    )
    model_kwargs = dict(
        simulated_delay=None,  # no RTC in the simulator
        max_token_len=96,
        memory_v5_oracle_writes=oracle_writes,
        memory_v5_prev_is_committed=not oracle_writes,  # B9: retry-until-committed; A9: plain change detector
        memory_v5_own_commit_label_content=False,  # own writes carry the model's own sentence
        memory_v5_write_delay_steps=0,
        memory_v5_prefill_history=True,
        memory_v5_prefill_max=16,
        memory_v5_reference_tokens=tokens,
        memory_state_mask_prob=0.5,
        # user 09-17 14:08: keep the bean 3-layer MLP bank but address it TOKEN AFTER TOKEN (the v6 contextual keys +
        # context-query pointer read of every run since 09-09) instead of the A8 slot-key template rule. This is the
        # v6.0 combination of 09-08: the probe of 09-09 (cluster_v6/README.md §7) recalled the NEWEST note per
        # context 71/71 on it but older notes 30/71, 17/71, 14/71 (the hidden layers pull context keys together);
        # the linear bank recalled 71/71 at every age. The count decisions here read only the newest note per slot.
        memory_v5_slot_keys=False,
        memory_v5_whiten_values=False,
        memory_v6_token_writes=True,
        memory_v6_whiten_keys=True,
        memory_v6_value_standardize=True,
        memory_v6_pointer_read=True,
        memory_v6_pointer_query="context",
        memory_v6_pointer_beta_init=10.0,
    )
    model_kwargs.update(model_overrides or {})
    model = dataclasses.replace(base_cfg.model, **model_kwargs)
    data = dataclasses.replace(
        _base_data(root, spec),
        base_config=dataclasses.replace(
            base_cfg.data.base_config,
            repo_id=REPO_ID,
            prompt_from_episode_meta=True,
            subtask_from_task=True,
            subtask_lookahead=0,
            action_target_offset_frames=1,
            memory_stride_frames=15,  # user 09-17 13:52: a memory step every 15 frames (2 Hz; beans ran 5, boba 15)
            memory_slice_prob=0.5,
            memory_min_slice_steps=14,
            memory_sequence_buckets=(14, 27, 40),
            evidence_subtasks=evidence,
            memory_required_subtasks=decision,
            memory_critical_prob=0.5,
            memory_critical_start_pad=75,
            memory_subtask_vocab=sentences,
            memory_waiting_state_dim=None,  # 8-D Panda state, no YAM anti-leak masking
            memory_episode_manifest_path=str(root / project_paths.ROBOMME_METADATA_DIR / TASK / "manifest.json"),
            memory_episode_manifest_sha256=spec["manifest_sha256"],
            memory_manifest_split="train",
            memory_manifest_split_seed=0,
            memory_v5_subtask_labels_path=str(root / project_paths.ROBOMME_METADATA_DIR / TASK / "subtasks_tgt.json"),
            memory_v5_subtask_labels_sha256=spec["subtasks_tgt_sha256"],
            memory_v5_generic_task=True,
            lerobot_dataset_root=str(root / spec["lerobot_dataset"]),
        ),
    )
    return dataclasses.replace(
        base_cfg,
        name=name,
        project_name="robomme_memory",
        model=model,
        data=data,
        assets_base_dir=str(root / project_paths.ROBOMME_ASSETS_ROOT),
        checkpoint_base_dir=str(root / project_paths.ROBOMME_CHECKPOINTS_DIR),
        batch_size=batch,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=peak_lr, decay_steps=10_000, decay_lr=peak_lr),
        weight_loader=weight_loaders.AuditedPartialCheckpointWeightLoader(
            loader_path,
            matched_allowlist=matched,
            fresh_init_allowlist=fresh,
            reinit_allowlist=(),
            ignored_source_allowlist=(),
            source_cast_dtype="float32",
        ),
        v4_graft_sources=(),
        num_train_steps=steps,
        save_interval=save_every,
        keep_period=keep,
        checkpoint_max_to_keep=max_keep,
        num_workers=8,
        fsdp_devices=fsdp,
        wandb_enabled=False,
        log_interval=10,  # user 09-20: losses every 10 updates, no diagnostic metrics
        log_diagnostics=False,
    )


def get_configs(existing: dict) -> list:
    root = project_paths.memory_project_root()
    ck = root / project_paths.ROBOMME_CHECKPOINTS_DIR
    plain_params = str(ck / f"pi05_robomme_base_{TASK}/legacy_plain_r1/7000/params")
    configs = []
    for tag, infix in _LABEL_TAGS.items():
        spec = _load_spec(root, tag)
        if spec is None:
            continue
        base_ki = _base_ki_config(root, spec, tag)
        base_ki_params = os.environ.get(
            "OPENPI_ROBOMME_BASE_PARAMS", str(ck / base_ki.name / f"pickxtimes_base_ki{infix}_r1/4999/params")
        )
        mem = f"pi05_robomme_mem_{TASK}{infix}"
        common = dict(matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=2001, save_every=500, keep=500,
                      peak_lr=5e-5, tag=tag)
        configs += [
            base_ki,
            # THE PLAN (09-15, official labels): boba2B on the KI base -- own write timing + label content, fresh memory
            # leaves, no stage A, no blinding.
            _mem_variant(root, spec, existing, f"{mem}_B", oracle_writes=False, loader_path=base_ki_params, **common),
            # boba5B port (memory_project_v7 cluster_v7/README.md §6, commits e697f52 + 32da1d3): the causal row that
            # PREDICTS the ordinal sees no image column at any block (raw text read at block 0, self-only in blocks
            # 1..8, self + the 8 sentence-bank slots after), so the count can only come from the bank/prompt. The
            # ordinal is one PaliGemma token; the repeat count sits in the prompt (no "of n"). Same schedule as _B.
            _mem_variant(
                root, spec, existing, f"{mem}_B_blind", oracle_writes=False, loader_path=base_ki_params,
                model_overrides=dict(memory_v7_digit_blind=True, memory_v7_digit_blind_patterns=_BLIND_PATTERNS[tag]),
                **common,
            ),
            # FULL SELF-WRITE (user 09-15 20:51 "enable full selfwrite, still use prefill, start from our existing 1k
            # ckpt"): own write timing AND own content (memory_v5_own_commit_label_content=False), analytic label
            # prefill kept, every parameter loaded from selfwrite r1 checkpoint 1000 (it already decodes the vocabulary
            # and fires writes at phase changes). Why: with label content the bank never holds the model's own wrong
            # ordinal in training, so it learned to copy the newest pick instead of adding one (ckpt 500/1000 traces).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_OWN_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001},
            ),
            # r3 candidate (09-15 23:40, prepared while r2 trains): own content + the A4 one-step write delay (the bank
            # never holds the current pick until one step after the onset, in training and at inference) + the pointer
            # copy bonus reset to 0 (r1 trained it to push the retrieved ordinal's logit; the third pick copied "second"
            # under that push). Init = r2/1000 by default (`OPENPI_ROBOMME_R3_INIT` overrides).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_delay", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R3_INIT", str(ck / f"{mem}_B_own/pickxtimes_off_own_r2/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1,
                                     memory_v6_pointer_beta_init=0.0),
                reinit=(r".*memory_v6_pointer_beta.*",),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001},
            ),
            # r3b (09-16 02:45): the write delay ALONE. r3's pointer beta reset to 0 never recovers (Adam at 5e-5 moves a
            # scalar ~0.05 per 1000 updates: r1 10.00->10.009 over 1000, r3 0->0.0066 at 500), so r3 lost the bank
            # read entirely (oracle ep7 10/35, "third"/"fifth" regardless of the bank). Same init r1/1000, beta kept.
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_delay_keepbeta", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R3B_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001},
            ),
            # r4 candidate (09-16 04:30, not launched): own content, beta kept, ONSET-WEIGHTED sentence CE (x10 on the
            # steps whose label differs from the previous step's). Why: under the label prefill ~13 of 14 rows per
            # pick are solved by copying the newest bank entry; the onset row alone needs "+1" (README 04:20).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_onset10", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R4_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v7_onset_ce_weight=10.0),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001},
            ),
            # r4b candidate (09-16 09:45, not launched): onset x10 AND the one-step write delay (r3b's lever), beta kept,
            # init r1/1000. Pairs with r4 to separate "onset weight alone" from "onset weight + delay".
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_onset10_delay", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R4_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v7_onset_ce_weight=10.0,
                                     memory_v5_write_delay_steps=1),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001},
            ),
            # r5 (user 09-16 11:57 "enable the gradual write ... keep the 1-step delay ... test with 0.3 first"): own content,
            # delay 1, beta kept, bank delta_rate 0.3 with a write on EVERY confident step (the prefill writes at 1.0).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_delay_grad03", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R5_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1,
                                     memory_v7_write_every_step=True, memory_v7_onset_ce_weight=10.0,  # user 12:03: onset x10 on both
                                     memory_semantic=dataclasses.replace(existing["pi05_yam_mem_v6_task1A2"].model.memory_semantic, delta_rate=0.3)),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001, "save_every": 200, "keep": 600, "max_keep": 2},  # user 12:11
            ),
            # r6 (user 09-16 11:58 "stop the 2xH100 run and run the debounce version"): own content, delay 1, beta kept,
            # commits only a sentence produced 2 steps in a row (the eval's self_debounce rule, now in training).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_delay_deb2", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R6_INIT", str(ck / f"{mem}_B_plain7000/pickxtimes_off_selfwrite_r1/1000/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1,
                                     memory_v7_write_debounce_steps=2, memory_v7_onset_ce_weight=10.0),  # user 12:03
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001, "save_every": 200, "keep": 600, "max_keep": 2},  # user 12:11
            ),
            # r7 candidate (09-16 15:30): the r6 recipe (own content, delay 1, debounce 2, onset x10) warm-started from
            # r6/600 (first fully correct five-pick count) plus the onset weight on the 2 steps BEFORE each label change,
            # against the early "pick k" announcements during place k-1 that caused r6's double counts (eps 1, 3).
            _mem_variant(
                root, spec, existing, f"{mem}_B_own_delay_deb2_pre2", oracle_writes=False,
                loader_path=os.environ.get("OPENPI_ROBOMME_R7_INIT", str(ck / f"{mem}_B_own_delay_deb2/pickxtimes_off_own_delay_deb2_r6/600/params")),
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1,
                                     memory_v7_write_debounce_steps=2, memory_v7_onset_ce_weight=10.0, memory_v7_onset_ce_pre_steps=2),
                **{**common, "matched": (r".+",), "fresh": (), "steps": 5001, "save_every": 200, "keep": 600, "max_keep": 2},
            ),
            # Literal "skip A and start from the existing base": the plain (no subtask head) checkpoint 7000.
            _mem_variant(root, spec, existing, f"{mem}_B_plain7000", oracle_writes=False, loader_path=plain_params, **common),
            # Oracle-write fallback (stage-A style) on the KI base, not launched.
            _mem_variant(root, spec, existing, f"{mem}_A", oracle_writes=True, loader_path=base_ki_params,
                         **{**common, "steps": 501, "save_every": 250, "keep": 250}),
        ]
        if tag == "official":
            # robomme_0916_v0 (user 09-17 00:01: "start fresh from the pi05 pretrain base ckpt and everything else as you
            # say ... use the 2h100"): the r6 recipe (own content, 1-step delay, 2-step debounce, label-history prefill,
            # linear delta bank, context pointer, official labels) from the PLAIN PickXtimes base 7000 with fresh memory
            # leaves (as r1), plus the generic v0 rules: flip-back retraction (3 steps), vocabulary-only writes, and the
            # per-token hard-word CE weight (x10) in place of the step-level onset weight. 5001 updates, save 200, keep the
            # newest 2 + multiples of 600. Run track: cluster_robomme/README.md "Run track".
            configs.append(_mem_variant(
                root, spec, existing, "pi05_robomme_0916_v0", oracle_writes=False, loader_path=plain_params,
                model_overrides=dict(memory_v5_own_commit_label_content=False, memory_v5_write_delay_steps=1,
                                     memory_v7_write_debounce_steps=2, memory_v7_write_retract_steps=3,
                                     memory_v7_write_vocab_only=True, memory_v7_hard_token_ce_weight=10.0),
                **{**common, "steps": 5001, "save_every": 200, "keep": 600, "max_keep": 2},
            ))
    spec_t = _load_spec(root, "tgt")
    if spec_t is not None:
        # robomme_0917_v1: the bean-scoop A9 -> B9 two-stage recipe on the target-carry labels (09-17). Stage A writes
        # the label sentences into the bank and trains the read/decoder on clean banks; stage B continues from A with
        # the model's own writes. OPENPI_ROBOMME_V1_GPUS (1 = the exact beans setup, batch 2 on one card; 2 = batch 4
        # on the H100 pair); OPENPI_ROBOMME_V1A_PARAMS picks the A checkpoint B starts from.
        gpus = int(os.environ.get("OPENPI_ROBOMME_V1_GPUS", "1"))
        a_params = os.environ.get("OPENPI_ROBOMME_V1A_PARAMS", str(ck / "pi05_robomme_0917_v1A/robomme_0917_v1A/1000/params"))
        configs.append(_beans_variant(root, spec_t, existing, "pi05_robomme_0917_v1A", oracle_writes=True, loader_path=plain_params,
                                      matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=2001, save_every=250, keep=500,
                                      peak_lr=5e-5, fsdp=gpus, batch=2 * gpus))
        configs.append(_beans_variant(root, spec_t, existing, "pi05_robomme_0917_v1B", oracle_writes=False, loader_path=a_params,
                                      matched=(r".+",), fresh=(), steps=2001, save_every=250, keep=500,
                                      peak_lr=2.5e-5, fsdp=gpus, batch=2 * gpus))
        # "_lin": the same bean recipe (labels, two stages, delay 0, retry, prefill 16, state mask 0.5, block 25, stride 15,
        # bean sampling) on the LINEAR v6.1 bank every run since 09-09 used (token-level whitened keys, context-query
        # pointer read) instead of the bean MLP bank + slot keys; none of the v7 knobs (user 09-17 13:52 Q1).
        bean_model = dict(memory_v5_write_delay_steps=0, memory_v5_prefill_max=16, memory_state_mask_prob=0.5,
                          memory_block_steps=25, memory_v5_own_commit_label_content=False,
                          memory_v7_write_debounce_steps=1, memory_v7_write_retract_steps=0, memory_v7_write_vocab_only=False,
                          memory_v7_hard_token_ce_weight=1.0, memory_v7_onset_ce_weight=1.0)
        bean_data = dict(memory_stride_frames=15, memory_slice_prob=0.5, memory_min_slice_steps=14,
                         memory_sequence_buckets=(14, 27, 40), memory_critical_prob=0.5, memory_critical_start_pad=75)
        a_lin_params = os.environ.get("OPENPI_ROBOMME_V1A_LIN_PARAMS", str(ck / "pi05_robomme_0917_v1A_lin/robomme_0917_v1A_lin/1000/params"))
        configs.append(_mem_variant(root, spec_t, existing, "pi05_robomme_0917_v1A_lin", oracle_writes=True, loader_path=plain_params,
                                    matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=2001, save_every=250, keep=500,
                                    peak_lr=5e-5, max_keep=2, tag="tgt", model_overrides=bean_model, data_overrides=bean_data))
        configs.append(_mem_variant(root, spec_t, existing, "pi05_robomme_0917_v1B_lin", oracle_writes=False, loader_path=a_lin_params,
                                    matched=(r".+",), fresh=(), steps=2001, save_every=250, keep=500,
                                    peak_lr=2.5e-5, max_keep=2, tag="tgt",
                                    model_overrides={**bean_model, "memory_v5_prev_is_committed": True}, data_overrides=bean_data))
        # "_cs" (09-17 20:12, user "yes do it"): B continued from B/500 with (1) COLD STARTS - about half of the training
        # clips begin at frame 0 with an empty bank instead of a quarter (full-trajectory mass = (1-critical)*(1-slice):
        # 0.7*0.7 = 0.49 vs 0.25), because the prompt-swap probe showed the target count x is copied from the bank rather
        # than read from the prompt; (2) onset CE weight 3 (the steps where the label changes count three times);
        # (3) first-note confirmation at serving: with an empty bank a sentence must be decoded twice in a row before
        # it is written (memory_v7_first_write_debounce_steps=2). Generic knobs only.
        b_lin_params = os.environ.get("OPENPI_ROBOMME_V1B_LIN_PARAMS", str(ck / "pi05_robomme_0917_v1B_lin/robomme_0917_v1B_lin/500/params"))
        configs.append(_mem_variant(root, spec_t, existing, "pi05_robomme_0917_v1B_lin_cs", oracle_writes=False, loader_path=b_lin_params,
                                    matched=(r".+",), fresh=(), steps=2001, save_every=250, keep=500,
                                    peak_lr=2.5e-5, max_keep=2, tag="tgt",
                                    model_overrides={**bean_model, "memory_v5_prev_is_committed": True, "memory_v7_onset_ce_weight": 3.0,
                                                     "memory_v7_first_write_debounce_steps": 2},
                                    data_overrides={**bean_data, "memory_slice_prob": 0.3, "memory_critical_prob": 0.3}))
    # 2026-09-18 03:43 (user): "stop the 2xH100 training, start a new pi05 with knowledge insulation and with the subtask,
    # put the subtask in the prompt so it follows the prompt, train to 10000, then the memory stages on top" + 03:46 "use
    # the official subtask". Base = the official-label KI recipe (`_base_ki_config`, continued from the plain 7000) plus the
    # v7 phase-context fields: the prompt carries the sentence of the PREVIOUS memory tick ("Last: <sentence>", stride 15
    # frames; dropout 0.1 -> "none" keeps a no-note route for tick 0 and after own mistakes). The KI head still has to
    # detect the change from the picture because the prompt shows the previous tick's sentence, not the current one; the
    # arm learns to follow a sentence in the place where the base model already follows text. Generic fields only.
    spec_o = _load_spec(root, "official")
    if spec_o is not None:
        ki_off = _base_ki_config(root, spec_o, "official")
        configs.append(dataclasses.replace(
            ki_off,
            name="pi05_robomme_0918_base_kiP_off",
            # +", Last:" + a 16-token slot + ";\n" over the KI 224. prompt_slot_len=16: the previous sentence enters as a
            # fixed-width slot of STANDALONE sentence ids (the same ids the causal head decodes and the sentence bank
            # stores), so the memory stages can fill the same slot from the bank (r2, 04:15; r1 used the text form).
            model=dataclasses.replace(ki_off.model, max_token_len=256, prompt_slot_len=16),
            data=dataclasses.replace(ki_off.data, base_config=dataclasses.replace(
                ki_off.data.base_config, prompt_prev_subtask=True, prompt_prev_subtask_stride=15, prompt_prev_subtask_dropout=0.1)),
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6),
            num_train_steps=10_000, save_interval=1_000, keep_period=2_000, checkpoint_max_to_keep=2, wandb_enabled=True,
        ))
    # 2026-09-18 03:54/03:57 (user): "once the base is ready start training A and B, our v2, try to make the model strongly
    # follow the prompt and subtask" + "fully remove the visual bank, do not inject, keep the tokens clean". v2 = the v1
    # bean recipe (linear token-key sentence bank, "Last:" read tokens = 8) on the OFFICIAL sentences, warm-started from the
    # new base (KI + prompt slot), with (1) the prompt slot: stage A fills it with the previous step's label (dropout 0.1 ->
    # "none"), stage B overwrites it with the model's own newest committed note (bank dropout 0.1); (2) no visual-bank
    # columns in the sequence (memory_v7_no_visual_block); (3) the B_cs training mix (49% cold starts, onset CE 3,
    # first-note confirmation). Generic knobs only; max_token_len 112 = context ~70 + ", Last:" + 16-slot + ";\n".
    if spec_o is not None:
        v2_base_params = os.environ.get("OPENPI_ROBOMME_V2_BASE_PARAMS",
                                        str(ck / "pi05_robomme_0918_base_kiP_off/robomme_0918_base_kiP_off_r2/9999/params"))
        v2_model = {**bean_model, "max_token_len": 112, "prompt_slot_len": 16, "prompt_slot_dropout": 0.1,
                    "memory_v7_no_visual_block": True}
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0918_v2A_off", oracle_writes=True, loader_path=v2_base_params,
                                    matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=501, save_every=250, keep=250,
                                    peak_lr=5e-5, max_keep=2, tag="official", model_overrides=v2_model, data_overrides=bean_data))
        v2a_params = os.environ.get("OPENPI_ROBOMME_V2A_PARAMS", str(ck / "pi05_robomme_0918_v2A_off/robomme_0918_v2A_off/500/params"))
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0918_v2B_off", oracle_writes=False, loader_path=v2a_params,
                                    matched=(r".+",), fresh=(), steps=2001, save_every=250, keep=500, peak_lr=2.5e-5, max_keep=2, tag="official",
                                    model_overrides={**v2_model, "memory_v5_prev_is_committed": True, "memory_v7_onset_ce_weight": 3.0,
                                                     "memory_v7_first_write_debounce_steps": 2,
                                                     "memory_v7_prompt_slot_from_bank": True, "memory_v7_prompt_slot_bank_dropout": 0.1},
                                    data_overrides={**bean_data, "memory_slice_prob": 0.3, "memory_critical_prob": 0.3}))
    # smoke-only (09-18 04:52): the v2B settings (own writes, slot overwritten from the bank, no visual block) but
    # warm-started like stage A (memory leaves fresh from the base), so the stage-B training path can be traced/compiled
    # on one card before A/500 exists. Never a real training config.
    if spec_o is not None:
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0918_v2B_off_smoke", oracle_writes=False, loader_path=v2_base_params,
                                    matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=3, save_every=1000, keep=1000, peak_lr=2.5e-5,
                                    max_keep=1, tag="official",
                                    model_overrides={**v2_model, "memory_v5_prev_is_committed": True, "memory_v7_onset_ce_weight": 3.0,
                                                     "memory_v7_first_write_debounce_steps": 2,
                                                     "memory_v7_prompt_slot_from_bank": True, "memory_v7_prompt_slot_bank_dropout": 0.1},
                                    data_overrides={**bean_data, "memory_slice_prob": 0.3, "memory_critical_prob": 0.3}))
    # 2026-09-19 03:30 (user): "on the 4 h200 i want to start v3 training, (1) action horizon set to 40 so predict next
    # 40 actions (2) memory tick every 5, so memory write at 4hz, and because we have 4h200 so you can set the window
    # larger and i also see it is very poor from second to 3rd or even more, so we can also consider train more on the
    # later pick up". v3 = the v2 recipe (official labels, KI base r2/9999 warm start, prompt slot, no visual block,
    # stage A label writes -> stage B own writes with the slot filled from the bank) with
    #   (1) action_horizon 40 (the plan is 40 steps = 2 s at the 20 Hz benchmark control);
    #   (2) memory tick every 5 frames (a note decision 4 times a second; v2: every 15 frames);
    #   (3) a longer window: V3_SEQ_STEPS ticks x 5 frames (160 -> 800 frames = 40 s; v2 covered 600 frames with 40 x 15;
    #       the median easy/medium demo is 449 frames, hard 812, so a cold start now runs a whole hard demo on its own
    #       notes) with the TBPTT fence every V3_BLOCK_STEPS=40 ticks (10 s) and 40/80/120/160 length buckets; the
    #       sampling mix 36% cold starts / 24% slices / 40% starts shortly before a note change (v2B: 49/21/30);
    #   (4) the generic later-history emphasis memory_start_history_power=1: inside the slice and the note-change
    #       branches a window that opens after n note changes is drawn (1+n) times as often (data_loader
    #       _history_power_reweight; counts changes, never reads the sentences), so the 2nd/3rd/... pick-ups, which
    #       only exist late in the long demos, get more of the training than their share of frames.
    # Everything else (bank, KI, prompt slot 16, onset CE 3 in B, first-note confirmation 2) as in v2. Batch, fsdp
    # devices and workers come from the launcher (robomme/logs/chain_v3_h200.sh: 4 x H200).
    if spec_o is not None:
        v3_model = {**v2_model, "action_horizon": 40, "memory_seq_steps": V3_SEQ_STEPS, "memory_block_steps": V3_BLOCK_STEPS}
        v3_data = {**bean_data, "memory_stride_frames": 5, "memory_min_slice_steps": 20,
                   "memory_sequence_buckets": tuple(range(40, V3_SEQ_STEPS + 1, 40)),
                   "memory_slice_prob": 0.4, "memory_critical_prob": 0.4, "memory_start_history_power": 1.0}
        v3b_model = {**v3_model, "memory_v5_prev_is_committed": True, "memory_v7_onset_ce_weight": 3.0,
                     "memory_v7_first_write_debounce_steps": 2,
                     "memory_v7_prompt_slot_from_bank": True, "memory_v7_prompt_slot_bank_dropout": 0.1}
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0919_v3A_off", oracle_writes=True, loader_path=v2_base_params,
                                    matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=V3_A_STEPS + 1, save_every=V3_A_STEPS // 2,
                                    keep=V3_A_STEPS // 2, peak_lr=5e-5, max_keep=2, tag="official", model_overrides=v3_model, data_overrides=v3_data))
        v3a_params = os.environ.get("OPENPI_ROBOMME_V3A_PARAMS",
                                    str(ck / f"pi05_robomme_0919_v3A_off/robomme_0919_v3A_off/{V3_A_STEPS}/params"))
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0919_v3B_off", oracle_writes=False, loader_path=v3a_params,
                                    matched=(r".+",), fresh=(), steps=V3_B_STEPS + 1, save_every=250, keep=500, peak_lr=2.5e-5, max_keep=2,
                                    tag="official", model_overrides=v3b_model, data_overrides=v3_data))
        # smoke-only: the B settings warm-started like A (memory leaves fresh from the base) so the B path can be traced
        # before A exists. Never a real training config.
        configs.append(_mem_variant(root, spec_o, existing, "pi05_robomme_0919_v3B_off_smoke", oracle_writes=False, loader_path=v2_base_params,
                                    matched=(NON_MEMORY_LEAF,), fresh=(MEMORY_LEAF,), steps=3, save_every=1000, keep=1000, peak_lr=2.5e-5,
                                    max_keep=1, tag="official", model_overrides=v3b_model, data_overrides=v3_data))
    return configs
