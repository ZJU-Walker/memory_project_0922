"""LED bean-scoop (real YAM station, dataset yam/bean_scoop_0905_v5) -- the real-robot memory policy line started 2026-09-22.

Two configs, run in sequence by beans/logs/chain_beans0922.sh:

  pi05_yam_beans0922_base  the plain pi0.5 base with knowledge insulation (the 2026-09-06 `pi05_yam_beans0905_base` recipe:
                           predict_subtask co-training on the 20 target-carry sentences, RTC delay 15, batch 16, lr 5e-5, EMA
                           0.999), trained to 10k updates (user 2026-09-21 23:52: "train a base policy first ... knowledge
                           insulation ... up to 10k steps"). Its checkpoints had been deleted, so it is retrained.
  pi05_yam_beans0922_v1    the 0920 v1 memory STRUCTURE (robomme_0920_config.V0_MODEL: 8 fixed read queries at the input, no
                           pointer / conditioner / slot, write every tick with no rule, own content with the label-write ramp,
                           onset CE 3, no past frames) on the beans v5 recipe (B9: tick 5 frames = 0.17 s at 30 Hz so every
                           LED blink is seen, 40-tick windows, TBPTT 25, buckets 14/27/40, slice 0.5 / anchored 0.5 / pad 75,
                           prefill 16, no state mask, horizon 50, max_token_len 80, lr 2.5e-5), warm-started from the base
                           10k with fresh memory leaves, 5000 updates, label-write ramp over the first 500 updates.

Portability: every path is project-root-relative (openpi.shared.project_paths; MEMORY_PROJECT_ROOT overrides the root) and
the machine-specific ones have environment overrides: OPENPI_BEANS_DATASET_ROOT (LeRobot dataset dir), OPENPI_BEANS_ASSETS_DIR
(norm stats), OPENPI_BEANS_PI05_BASE (pi05_base params for the base run; default the public gs:// checkpoint),
OPENPI_BEANS_BASE_PARAMS (the base checkpoint the memory run starts from), OPENPI_BEANS_BATCH / OPENPI_BEANS_BASE_BATCH.
"""

import dataclasses
import os

from openpi.shared import project_paths
from openpi.training import config as cfg
from openpi.training import optimizer, weight_loaders
from openpi.training import robomme_0920_config as _r0920

TASK = "beans0922"
DATASET_ROOT_REL = "v5/data/lerobot/yam/bean_scoop_0905_v5"  # 89 episodes / 71089 frames, rebuilt 2026-09-21 from data/0905beans_*
ASSETS_DIR_REL = "v5/assets/pi05_yam_bean_scoop_0905_v5"  # norm stats of that dataset (repo id yam/bean_scoop_0905_v5)
CHECKPOINTS_REL = "beans/checkpoints"
BASE_EXP = "beans0922_base"
BASE_STEPS = 10_000
BASE_SAVE_EVERY = 2_500
BASE_BATCH = int(os.environ.get("OPENPI_BEANS_BASE_BATCH", "16"))
MEM_STEPS = 5_000
MEM_LABEL_WRITE_STEPS = 500  # label-write probability 1 -> 0 over the first 500 updates, fully self-written after (user 09-22 00:36)
MEM_BATCH = int(os.environ.get("OPENPI_BEANS_BATCH", "4"))  # 2 windows per 80 GB card at fsdp 2; the launcher falls back to 2
MEM_PEAK_LR = 2.5e-5  # beans B9
FSDP = 2
WORKERS = 12

# the 0920 v1 structure = V0_MODEL minus the RoboMME window / horizon / camera / regularisation choices (those come from beans v5)
_KEEP_TASK = {"action_horizon", "max_token_len", "memory_seq_steps", "memory_block_steps", "memory_state_mask_prob",
              "memory_v5_prefill_max", "memory_v0920_history_frames", "memory_v0920_history_pool", "memory_v0920_history_dropout"}
STRUCTURE = {k: v for k, v in _r0920.V0_MODEL.items() if k not in _KEEP_TASK}
STRUCTURE.update(memory_v0920_history_frames=0, memory_v0920_history_dropout=0.0, memory_v0920_drop_blank_camera=False)


def _root():
    return project_paths.memory_project_root()


def dataset_root() -> str:
    return os.environ.get("OPENPI_BEANS_DATASET_ROOT") or str(_root() / DATASET_ROOT_REL)


def assets_dir() -> str:
    return os.environ.get("OPENPI_BEANS_ASSETS_DIR") or str(_root() / ASSETS_DIR_REL)


def base_params_path() -> str:
    return os.environ.get("OPENPI_BEANS_BASE_PARAMS") or str(_root() / CHECKPOINTS_REL / "pi05_yam_beans0922_base" / BASE_EXP / str(BASE_STEPS) / "params")


def _with_beans_data(data):
    return dataclasses.replace(
        data, repo_id="yam/bean_scoop_0905_v5",
        base_config=dataclasses.replace(data.base_config, lerobot_dataset_root=dataset_root()),
        assets=cfg.AssetsConfig(assets_dir=assets_dir()),
    )


def base_config(existing: dict, name: str = "pi05_yam_beans0922_base", *, steps: int = BASE_STEPS, wandb: bool = True) -> cfg.TrainConfig:
    """The 09-06 knowledge-insulation base recipe, 10k updates, on the 2-card pair (batch 16 = 8 per card, FSDP 2)."""
    src = existing["pi05_yam_beans0905_base"]
    loader = weight_loaders.CheckpointWeightLoader(os.environ.get("OPENPI_BEANS_PI05_BASE", "gs://openpi-assets/checkpoints/pi05_base/params"))
    return dataclasses.replace(
        src, name=name, data=_with_beans_data(src.data), weight_loader=loader,
        checkpoint_base_dir=str(_root() / CHECKPOINTS_REL), assets_base_dir=str(_root() / "beans/assets"),
        num_train_steps=steps + 1, save_interval=BASE_SAVE_EVERY, keep_period=BASE_SAVE_EVERY, checkpoint_max_to_keep=2,
        batch_size=BASE_BATCH, fsdp_devices=FSDP, num_workers=WORKERS,
        project_name=TASK, wandb_enabled=wandb, log_interval=10, log_diagnostics=False,
    )


# v2 write rule (user 09-22 14:24, after the checkpoint-1500 held-out probe): a note is written only when the decoded sentence
# differs from the last COMMITTED one (memory_v5_prev_is_committed stays True = retry until committed) AND its mean token
# probability is >= 0.9 (B9's gate); label writes during the ramp are always confident. v1 wrote every tick with no gate, so a
# wrong "scoop 2 times" was rewritten 15x and read back (2 -> 3 drift on dev ep 29/73) and junk decodes entered the bank.
V2_WRITE_RULE = dict(memory_v7_write_every_step=False, memory_v5_write_conf=0.9)


def memory_config(existing: dict, name: str = "pi05_yam_beans0922_v1", *, steps: int = MEM_STEPS, wandb: bool = True,
                  batch: int = MEM_BATCH, model_overrides: dict | None = None) -> cfg.TrainConfig:
    """v1 structure on the beans v5 (B9) window / labels / sampling, from the beans0922 base with fresh memory leaves.
    `model_overrides` = the flags a later revision changes on top of v1 (v2: V2_WRITE_RULE)."""
    template = existing["pi05_yam_mem_v6_task1A2"]  # the linear delta-rule bank template every 0920 config derives from
    b9 = existing["pi05_yam_mem_v5_beansB9"]  # the beans v5 recipe: data, labels, window, reference tokens
    model_kwargs = dict(STRUCTURE)  # the v1 structure (includes prefill_history True, own writes, ramp-compatible flags)
    model_kwargs.update(
        action_horizon=b9.model.action_horizon,  # 50
        max_token_len=b9.model.max_token_len,  # 80
        simulated_delay=15,  # RTC budget of the base (B9 trained with 6, its successors A10/B10 raised it to 15)
        memory_seq_steps=b9.model.memory_seq_steps,  # 40 ticks x 5 frames = 200 frames
        memory_block_steps=b9.model.memory_block_steps,  # 25
        memory_v5_prefill_max=b9.model.memory_v5_prefill_max,  # 16
        memory_v5_reference_tokens=b9.model.memory_v5_reference_tokens,  # the 20 target-carry sentences
        memory_state_mask_prob=0.0,  # no state masking (user 09-22 00:36; B9 used 0.5)
    )
    model_kwargs.update(model_overrides or {})
    model = dataclasses.replace(template.model, **model_kwargs)
    data = _with_beans_data(b9.data)
    return dataclasses.replace(
        template, name=name, model=model, data=data,
        checkpoint_base_dir=str(_root() / CHECKPOINTS_REL), assets_base_dir=str(_root() / "beans/assets"),
        weight_loader=weight_loaders.AuditedPartialCheckpointWeightLoader(
            base_params_path(), matched_allowlist=(cfg._V7_NON_MEMORY_LEAF,), fresh_init_allowlist=(cfg._V7_MEMORY_LEAF,),  # noqa: SLF001
            reinit_allowlist=(), ignored_source_allowlist=(), source_cast_dtype="float32",
        ),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=MEM_PEAK_LR, decay_steps=10_000, decay_lr=MEM_PEAK_LR),
        num_train_steps=steps + 1, save_interval=250, keep_period=1_000, checkpoint_max_to_keep=2,
        batch_size=batch, fsdp_devices=FSDP, num_workers=WORKERS, label_write_schedule_steps=MEM_LABEL_WRITE_STEPS,
        project_name=TASK, wandb_enabled=wandb, log_interval=10, log_diagnostics=False,
    )


def get_configs(existing: dict) -> list:
    return [
        base_config(existing),
        base_config(existing, "pi05_yam_beans0922_base_smoke", steps=2, wandb=False),
        memory_config(existing),
        memory_config(existing, "pi05_yam_beans0922_v1_smoke", steps=2, wandb=False),
        memory_config(existing, "pi05_yam_beans0922_v2", model_overrides=V2_WRITE_RULE),
        memory_config(existing, "pi05_yam_beans0922_v2_smoke", steps=2, wandb=False, model_overrides=V2_WRITE_RULE),
    ]
