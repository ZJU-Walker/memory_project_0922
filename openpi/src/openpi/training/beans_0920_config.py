"""Bean scoop (YAM boba station, dataset boba_0913_v2) with the 0920 v1 memory STRUCTURE and the bean-scoop recipe for
everything else (user 2026-09-21 22:46: "apply current setup to our real world task the bean scoop task ... start from
pretrained pi05 scoop base ... use 2h100 ... for ticks lets use the current bean scoop setup, only change the structure here,
other thing keep same as the bean scoop").

Structure (robomme_0920_config.V0_MODEL, as trained in pi05_robomme_0920_v1): 8 fixed learned read queries at the INPUT,
visible to all 18 blocks; no pointer bonus, no read conditioner, no "Last:" prompt slot, no visual bank block; write every
tick with no rule, own content, label content with probability 1 -> 0 over the first 30 % of updates
(TrainConfig.label_write_schedule_steps); token CE with onset weight 3; no past frames (the station's three real cameras stay).

Kept from the bean-scoop recipe (config.py _v7_boba_mem_variant / boba2B, boba3B data): boba base 9999
(v6 pi05_yam_boba0911_base, plain pi05 fine-tune, RTC delay 15) with every memory leaf fresh; two-phase labels v3 on
boba_0913_v2 (56 episodes + 30 empty-scoop clips, 25 sentences, prefill 26); tick 15 frames (0.5 s at 30 Hz), 60-tick
windows (30 s), TBPTT 30, buckets (20, 40, 60), slice 0.9 / critical 0.5 / pad 75 / min slice 14; action horizon 50,
max_token_len 80, state mask 0; lr 5e-5 (warmup 100, then constant), AdamW clip 1, memory grad clip 5; save 250 / keep 500.
"""

import dataclasses
import os

from openpi.shared import project_paths
from openpi.training import config as cfg
from openpi.training import robomme_0920_config as _r0920

BOBA_BASE = "v6/checkpoints/pi05_yam_boba0911_base/pi05_boba0911_base_rtc15_20260912_r1/9999/params"
BOBA3_SIDECAR = "boba_v5_subtask_labels_v3.json"  # two-phase labels for the 56 episodes + the 30 empty-scoop clips (v7 09-14)
BOBA3_SIDECAR_SHA256 = "a76a1de90a000682433e8e190f477c9c25be8c85e8b6932c14ee3b4999f7b352"
BOBA3_MANIFEST = "boba_episode_manifest_v2.json"
BOBA3_MANIFEST_SHA256 = "161833a4bd3b6fc689db4efc96370a44cd60e7d60e0a4287d151f4e21d46cc9c"
BOBA3_REPO_ID = "yam/boba_0913_v2"
BOBA3_LEROBOT_ROOT = "v6/data/lerobot/yam/boba_0913_v2"

BEANS_STEPS = 3_000  # boba2B ran 2000 + 1500 continuation; one run here
BEANS_LABEL_WRITE_STEPS = 900  # label-write probability 1 -> 0 over the first 30 % (0920: 1500 of 5000)
BEANS_BATCH = int(os.environ.get("OPENPI_BEANS_BATCH", "4"))  # 2 windows per H100 at fsdp 2 (boba2B: 2 per H200); launcher falls back to 2
BEANS_WORKERS = 12
BEANS_PEAK_LR = 5e-5

# the 0920 v1 structure = V0_MODEL minus the RoboMME window / horizon / camera / regularisation choices (those stay boba)
_KEEP_BOBA = {"action_horizon", "max_token_len", "memory_seq_steps", "memory_block_steps", "memory_state_mask_prob",
              "memory_v5_prefill_max", "memory_v0920_history_frames", "memory_v0920_history_pool", "memory_v0920_history_dropout"}
# set by _v7_boba_mem_variant itself (a duplicate keyword would raise); re-applied after it
_AFTER_KEYS = ("memory_v5_oracle_writes", "memory_v5_prev_is_committed", "memory_v5_own_commit_label_content")
STRUCTURE = {k: v for k, v in _r0920.V0_MODEL.items() if k not in _KEEP_BOBA and k not in _AFTER_KEYS}
STRUCTURE.update(memory_v0920_history_frames=0, memory_v0920_history_dropout=0.0, memory_v0920_drop_blank_camera=False)
AFTER = {k: _r0920.V0_MODEL[k] for k in _AFTER_KEYS}  # oracle False, prev committed True, own content (label only via the ramp)


def _beans_v1(name: str, *, steps: int, save_every: int, keep: int, wandb: bool, batch: int = BEANS_BATCH) -> cfg.TrainConfig:
    root = project_paths.memory_project_root()
    base = cfg._v7_boba_mem_variant(  # noqa: SLF001
        name, oracle_writes=False, loader_path=BOBA_BASE, matched=(cfg._V7_NON_MEMORY_LEAF,), fresh=(cfg._V7_MEMORY_LEAF,),  # noqa: SLF001
        steps=steps, save_every=save_every, keep=keep, peak_lr=BEANS_PEAK_LR, model_overrides=STRUCTURE,
        **{**cfg._V7_BOBA2, "sidecar_name": BOBA3_SIDECAR, "sidecar_sha256": BOBA3_SIDECAR_SHA256},  # noqa: SLF001
    )
    boba_dir = root / "openpi/cluster_v7/boba"
    data = dataclasses.replace(
        base.data, repo_id=BOBA3_REPO_ID,
        base_config=dataclasses.replace(
            base.data.base_config,
            memory_episode_manifest_path=str(boba_dir / BOBA3_MANIFEST), memory_episode_manifest_sha256=BOBA3_MANIFEST_SHA256,
            lerobot_dataset_root=os.environ.get("OPENPI_BOBA2_LEROBOT_ROOT") or str(root / BOBA3_LEROBOT_ROOT),
        ),
    )
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, **AFTER),
        data=data,
        checkpoint_base_dir=str(root / "beans/checkpoints"),
        batch_size=batch,
        fsdp_devices=2,
        num_workers=BEANS_WORKERS,
        label_write_schedule_steps=BEANS_LABEL_WRITE_STEPS,
        checkpoint_max_to_keep=2,
        project_name="beans_0920",
        wandb_enabled=wandb,
        log_interval=10,
        log_diagnostics=False,
    )


def get_configs() -> list:
    return [
        _beans_v1("pi05_yam_beans_0920_v1", steps=BEANS_STEPS + 1, save_every=250, keep=500, wandb=True),
        _beans_v1("pi05_yam_beans_0920_v1_smoke", steps=3, save_every=1_000, keep=1_000, wandb=False),
    ]
