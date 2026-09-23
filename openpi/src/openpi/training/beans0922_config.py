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
# v3 (user 09-22 14:37 "ok do it"): v2 + "look before you ask" -- each read question is shifted by a pooled summary of the
# tick's input tokens and by the embedding of the last committed note (both through zero-initialised maps), the answers
# still enter at the input (Pi0Config.memory_v0920_query_context).
# + the error-driven token weight (user 09-22 15:18 "lets use the threshold"): sentence tokens whose own teacher-forced
# prediction is wrong at that tick weigh 5x in the sentence CE (the count word is 1 token in ~50 per tick; the weight fades
# once the word is learned) and the wrong-token telemetry v7_hard_token_count switches on.
V3_QUERY_CONTEXT = dict(V2_WRITE_RULE, memory_v0920_query_context=True, memory_v7_hard_token_ce_weight=5.0)
# v4 "token bank, exact copies" (user 09-23 23:55 "ok开始做", after the geometry probes on v3/1000 and v1/2000): the token
# store and the writes were measured to be fine (a stored digit reads back at cosine 1.00 with its exact context key); what
# failed was the READ -- (a) v3's question shift was 170x the base questions and rank-one, so all 8 questions collapsed into
# one; (b) an answer is a blend over the whole history in which the go note fades (1 %/tick: 15 % strength by the third
# scoop note, cosine to the true digit 0.71) and the newest note dominates, so "of x" was a noisy copy that flipped and
# then stayed wrong. v4 = v2's write rule + the 5x error-driven token weight, WITHOUT the shift, plus:
#   * memory_v6_pointer_read in "context" mode (beta 10, the v6.1 setting): while a token is decoded the bank is asked
#     "what followed this exact context last time" and the answer is added to the token scores -- "scoop 2 of _" fetches
#     the previous x exactly, "scoop _" the previous k; copies stop flipping;
#   * memory_v0920_prev_readback: the last committed note is read back through the bank with its own keys and enters the
#     input token by token (exact), so the tray decision (k == x -> done) sees both digits instead of a blend;
#   * bank decay 0.01 -> 0.001 per tick (half-life 115 s instead of 69 ticks = 12 s; A6sd precedent): the go note stays
#     readable through the whole scoop phase; with change-only writes (~15 notes per episode) nothing needs forgetting;
#   * the write gate compares the LOWEST token probability (memory_v5_write_conf_min) with 0.8 instead of the mean with 0.9:
#     one doubtful word keeps a note out; the mean gate never fired at a wrong count.
# Still learned, not exact: the yellow-go count (context "scoop, scoop _" is new) and the first scoop note's x ("scoop 1
# of _" is new) come from the 8 fixed questions reading the fresh light-off / go note.
V4_TOKEN_EXACT = dict(
    V2_WRITE_RULE, memory_v7_hard_token_ce_weight=5.0,
    memory_v6_pointer_read=True, memory_v6_pointer_query="context", memory_v6_pointer_beta_init=10.0,
    memory_v0920_prev_readback=True,
    memory_v5_write_conf_min=True, memory_v5_write_conf=0.8,
)
V4_BANK = dict(alpha_step=0.001)
# v4b (user 09-23 05:18 "ok do it", after the 500 / 750 own-note probes): the 0.8 minimum-probability gate starved the bank --
# the model's own light-phase sentences sit at 0.4-0.5, so after the label ramp almost no own note was written in training, the
# model learned to guess the count without notes and (by 750) to override the notes it had (27 writes flickering "of 2" /
# "of 3" on ep 25; "scoop 2 times" at 0.98 on the one-blink ep 29). Teacher-forced metrics did not show it because most
# windows start with the LABEL history prefilled. v4b = v4 resumed from checkpoint 500 with the write rule the model can
# actually pass: changed vs the last committed note, every word above 0.3 ("no word is very unlikely"), and decoded the same
# way on two consecutive ticks (memory_v7_write_debounce_steps = 2, the generic v7 rule; the video script's plain self mode
# and serve_yam_memory.py already honour it). The same rule in training and at deployment.
V4B_WRITE_RULE = dict(V4_TOKEN_EXACT, memory_v5_write_conf=0.3, memory_v7_write_debounce_steps=2)
# v4c (prepared 09-23 07:45 after the v4b/750 own-note probe; launched only if 1000 confirms): with v4b the light notes are
# written and right, but the go ONSET ignores them ("scoop 2 times" at 0.93 with "light off: 1" read back exactly, eps 29/73)
# and mid-phase the model overrides its own correct note when the picture is ambiguous (ep 25: "3 times" / "done" while the
# arm reaches for the scoop). Cause: between 500 and 700 the model trained on its own light notes while they were wrong
# 10-25 % of the time, so the note became an unreliable feature at the onset and the model fell back on the picture / the
# prior; the onset is one tick per window, so the teacher-forced curves never showed it. Two existing generic mechanisms:
#   * memory_v5_own_commit_label_content=True (v6.2, the B2 lesson): the model still decides WHEN to write (change / gate /
#     two ticks), but what enters the bank in training is the label sentence of that tick, so the bank never contradicts the
#     targets and the note stays a reliable feature -- reliance on the note is what we want the model to learn;
#   * memory_v7_onset_ce_weight 3 -> 6: more of the sentence loss on the ticks where the sentence changes (the onset), the
#     only ticks that cannot be solved by copying the newest note.
V4C_TRUST_NOTES = dict(V4B_WRITE_RULE, memory_v5_own_commit_label_content=True, memory_v7_onset_ce_weight=6.0)


def memory_config(existing: dict, name: str = "pi05_yam_beans0922_v1", *, steps: int = MEM_STEPS, wandb: bool = True,
                  batch: int = MEM_BATCH, model_overrides: dict | None = None, bank_overrides: dict | None = None) -> cfg.TrainConfig:
    """v1 structure on the beans v5 (B9) window / labels / sampling, from the beans0922 base with fresh memory leaves.
    `model_overrides` = the flags a later revision changes on top of v1 (v2: V2_WRITE_RULE); `bank_overrides` = fields of
    the sentence bank's MemoryConfig a revision changes (v4: the decay alpha_step)."""
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
    if bank_overrides:
        # both banks: Pi0Config requires memory_semantic.alpha_step == memory.alpha_step (the visual bank is inert in the
        # 0920 design -- no visual columns -- but its config must agree)
        model_kwargs["memory_semantic"] = dataclasses.replace(template.model.memory_semantic, **bank_overrides)
        model_kwargs["memory"] = dataclasses.replace(template.model.memory, **bank_overrides)
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
        memory_config(existing, "pi05_yam_beans0922_v3", model_overrides=V3_QUERY_CONTEXT),
        memory_config(existing, "pi05_yam_beans0922_v3_smoke", steps=2, wandb=False, model_overrides=V3_QUERY_CONTEXT),
        memory_config(existing, "pi05_yam_beans0922_v4", model_overrides=V4_TOKEN_EXACT, bank_overrides=V4_BANK),
        memory_config(existing, "pi05_yam_beans0922_v4_smoke", steps=2, wandb=False, model_overrides=V4_TOKEN_EXACT,
                      bank_overrides=V4_BANK),
        memory_config(existing, "pi05_yam_beans0922_v4b", model_overrides=V4B_WRITE_RULE, bank_overrides=V4_BANK),
        memory_config(existing, "pi05_yam_beans0922_v4b_smoke", steps=2, wandb=False, model_overrides=V4B_WRITE_RULE,
                      bank_overrides=V4_BANK),
        memory_config(existing, "pi05_yam_beans0922_v4c", model_overrides=V4C_TRUST_NOTES, bank_overrides=V4_BANK),
        memory_config(existing, "pi05_yam_beans0922_v4c_smoke", steps=2, wandb=False, model_overrides=V4C_TRUST_NOTES,
                      bank_overrides=V4_BANK),
    ]
