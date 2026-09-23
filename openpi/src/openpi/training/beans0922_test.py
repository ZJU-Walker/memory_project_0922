"""beans0922: the base is the 09-06 KI recipe at 10k on two cards; the memory run is the 0920 v1 structure (== the RoboMME v1
config flag for flag) on the beans v5 window / labels, warm-started from that base with fresh memory leaves."""
import dataclasses
import pathlib

import pytest


def test_beans0922_configs():
    from openpi.training import config as _config
    from openpi.training import beans0922_config as b

    root = pathlib.Path(__file__).resolve().parents[4]
    base, mem = _config.get_config("pi05_yam_beans0922_base"), _config.get_config("pi05_yam_beans0922_v1")
    src, b9 = _config.get_config("pi05_yam_beans0905_base"), _config.get_config("pi05_yam_mem_v5_beansB9")
    # base: the 09-06 recipe, only length / cards / paths changed
    assert base.model == src.model and base.num_train_steps == 10_001 and (base.batch_size, base.fsdp_devices) == (16, 2)
    assert base.lr_schedule == src.lr_schedule and base.ema_decay == src.ema_decay == 0.999 and base.optimizer == src.optimizer
    assert base.data.base_config.memory_v5_subtask_labels_path == src.data.base_config.memory_v5_subtask_labels_path
    assert base.data.base_config.lerobot_dataset_root == b.dataset_root() and base.data.assets.assets_dir == b.assets_dir()
    assert base.checkpoint_base_dir == str(root / "beans/checkpoints") and base.project_name == "beans0922"
    # memory: structure == RoboMME v1
    if (root / "robomme/metadata/PickXtimes/prepared_official.json").is_file():
        v1 = _config.get_config("pi05_robomme_0920_v1")
        for k in b.STRUCTURE:
            if k != "memory_v0920_drop_blank_camera":
                assert getattr(mem.model, k) == getattr(v1.model, k), k
    assert mem.model.memory_v0920_input_read and mem.model.memory_v7_write_every_step and not mem.model.memory_v6_pointer_read
    assert (mem.model.memory_v5_oracle_writes, mem.model.memory_v5_prev_is_committed, mem.model.memory_v5_own_commit_label_content) == (False, True, False)
    assert mem.model.memory_v0920_history_frames == 0 and not mem.model.memory_v0920_drop_blank_camera
    # memory: window / labels / sampling == beans v5 B9
    for k in ("action_horizon", "max_token_len", "memory_seq_steps", "memory_block_steps", "memory_v5_prefill_max", "memory_v5_reference_tokens"):
        assert getattr(mem.model, k) == getattr(b9.model, k), k
    assert (mem.model.action_horizon, mem.model.memory_seq_steps, mem.model.memory_block_steps, mem.model.simulated_delay) == (50, 40, 25, 15)
    d, db = mem.data.base_config, b9.data.base_config
    for k in ("memory_stride_frames", "memory_min_slice_steps", "memory_sequence_buckets", "memory_slice_prob", "memory_critical_prob",
              "memory_critical_start_pad", "memory_subtask_vocab", "evidence_subtasks", "memory_required_subtasks", "memory_waiting_state_dim",
              "memory_v5_subtask_labels_path", "memory_v5_subtask_labels_sha256", "memory_episode_manifest_path", "memory_manifest_split", "subtask_lookahead"):
        assert getattr(d, k) == getattr(db, k), k
    assert (d.memory_stride_frames, d.memory_sequence_buckets, len(d.memory_subtask_vocab)) == (5, (14, 27, 40), 20)
    assert d.lerobot_dataset_root == b.dataset_root() and mem.data.assets.assets_dir == b.assets_dir()
    # warm start from the beans0922 base, memory leaves fresh; B9 schedule; ramp
    assert mem.weight_loader.params_path == b.base_params_path() and mem.weight_loader.params_path.endswith("beans0922_base/10000/params")
    assert mem.weight_loader.fresh_init_allowlist and mem.lr_schedule == b9.lr_schedule
    assert (mem.batch_size, mem.fsdp_devices, mem.num_train_steps, mem.label_write_schedule_steps) == (b.MEM_BATCH, 2, 5_001, 500)
    assert mem.model.memory_state_mask_prob == 0.0
    assert mem.memory_grad_clip == 5.0 and mem.freeze_filter == b9.freeze_filter
    spec_obs, spec_act = mem.model.inputs_spec(batch_size=1)
    assert sorted(spec_obs.images) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"] and spec_act.shape[-2] == 50
    for n in ("pi05_yam_beans0922_base_smoke", "pi05_yam_beans0922_v1_smoke"):
        s = _config.get_config(n); assert s.num_train_steps == 3 and not s.wandb_enabled


def test_v2_is_v1_with_the_change_only_confident_write_rule():
    from openpi.training import config as _config

    v1 = _config.get_config("pi05_yam_beans0922_v1")
    v2 = _config.get_config("pi05_yam_beans0922_v2")
    assert v2.model.memory_v7_write_every_step is False and v1.model.memory_v7_write_every_step is True
    assert v2.model.memory_v5_write_conf == 0.9 and v1.model.memory_v5_write_conf == 0.0
    assert v2.model.memory_v5_prev_is_committed is True
    changed = {f.name for f in dataclasses.fields(v1.model) if getattr(v1.model, f.name) != getattr(v2.model, f.name)}
    assert changed == {"memory_v7_write_every_step", "memory_v5_write_conf"}
    for field in ("data", "weight_loader", "lr_schedule", "num_train_steps", "batch_size", "label_write_schedule_steps"):
        assert getattr(v1, field) == getattr(v2, field), field


def test_v3_is_v2_with_the_question_context():
    from openpi.training import config as _config

    v2 = _config.get_config("pi05_yam_beans0922_v2")
    v3 = _config.get_config("pi05_yam_beans0922_v3")
    assert v3.model.memory_v0920_query_context is True and v2.model.memory_v0920_query_context is False
    changed = {f.name for f in dataclasses.fields(v2.model) if getattr(v2.model, f.name) != getattr(v3.model, f.name)}
    assert changed == {"memory_v0920_query_context", "memory_v7_hard_token_ce_weight"}
    assert v3.model.memory_v7_hard_token_ce_weight == 5.0 and v2.model.memory_v7_hard_token_ce_weight == 1.0
    assert v3.model.memory_v7_write_every_step is False and v3.model.memory_v5_write_conf == 0.9


def test_v4_is_v2_with_exact_copies_and_a_slow_bank():
    """v4 (09-23): v2's write rule + the 5x token weight, no question shift, pointer read (context, beta 10), the last note
    read back at the input, the lowest-token-probability gate at 0.8, bank decay 0.001."""
    from openpi.training import config as _config

    v2 = _config.get_config("pi05_yam_beans0922_v2")
    v3 = _config.get_config("pi05_yam_beans0922_v3")
    v4 = _config.get_config("pi05_yam_beans0922_v4")
    m = v4.model
    assert (m.memory_v6_pointer_read, m.memory_v6_pointer_query, m.memory_v6_pointer_beta_init) == (True, "context", 10.0)
    assert m.memory_v0920_prev_readback is True and m.memory_v0920_query_context is False
    assert (m.memory_v5_write_conf_min, m.memory_v5_write_conf, m.memory_v7_write_every_step) == (True, 0.8, False)
    assert m.memory_v7_hard_token_ce_weight == 5.0 and m.memory_v6_token_writes and m.memory_v6_whiten_keys
    assert m.memory_semantic.alpha_step == 0.001 and v2.model.memory_semantic.alpha_step == 0.01
    assert dataclasses.replace(m.memory_semantic, alpha_step=0.01) == v2.model.memory_semantic
    changed = {f.name for f in dataclasses.fields(v2.model) if getattr(v2.model, f.name) != getattr(m, f.name)}
    # (the v6.1 template already carries pointer_query="context" and beta_init 10; v0 only switched the pointer off)
    assert changed == {"memory_v6_pointer_read", "memory_v0920_prev_readback", "memory_v5_write_conf_min", "memory_v5_write_conf",
                       "memory_v7_hard_token_ce_weight", "memory_semantic", "memory"}
    assert m.memory.alpha_step == 0.001 and dataclasses.replace(m.memory, alpha_step=0.01) == v2.model.memory
    assert v3.model.memory_v0920_query_context is True  # v4 does not inherit the shift
    for field in ("data", "weight_loader", "lr_schedule", "num_train_steps", "batch_size", "label_write_schedule_steps"):
        assert getattr(v2, field) == getattr(v4, field), field
    # the read-back tokens widen the memory block by one token per note position
    assert v4.model.memory_v5_sentence_len == v2.model.memory_v5_sentence_len
    s = _config.get_config("pi05_yam_beans0922_v4_smoke"); assert s.num_train_steps == 3 and not s.wandb_enabled


def test_v4b_is_v4_with_the_passable_write_gate():
    """v4b (09-23 05:18): v4 resumed from checkpoint 500 with the write gate the model's own sentences can pass -- lowest word
    probability >= 0.3 and the same sentence on two consecutive ticks; nothing else changes."""
    from openpi.training import config as _config

    v4 = _config.get_config("pi05_yam_beans0922_v4")
    v4b = _config.get_config("pi05_yam_beans0922_v4b")
    assert (v4b.model.memory_v5_write_conf, v4b.model.memory_v7_write_debounce_steps) == (0.3, 2)
    assert (v4.model.memory_v5_write_conf, v4.model.memory_v7_write_debounce_steps) == (0.8, 1)
    assert v4b.model.memory_v5_write_conf_min and v4b.model.memory_v0920_prev_readback and v4b.model.memory_v6_pointer_read
    changed = {f.name for f in dataclasses.fields(v4.model) if getattr(v4.model, f.name) != getattr(v4b.model, f.name)}
    assert changed == {"memory_v5_write_conf", "memory_v7_write_debounce_steps"}
    for field in ("data", "weight_loader", "lr_schedule", "num_train_steps", "batch_size", "label_write_schedule_steps"):
        assert getattr(v4, field) == getattr(v4b, field), field
    s = _config.get_config("pi05_yam_beans0922_v4b_smoke"); assert s.num_train_steps == 3 and not s.wandb_enabled


def test_v4c_is_v4b_that_trusts_its_notes():
    """v4c (prepared 09-23 07:45): v4b + label content on own-write commits (the bank never contradicts the targets) and the
    onset weight 3 -> 6; nothing else changes."""
    from openpi.training import config as _config

    v4b = _config.get_config("pi05_yam_beans0922_v4b")
    v4c = _config.get_config("pi05_yam_beans0922_v4c")
    assert v4c.model.memory_v5_own_commit_label_content is True and v4b.model.memory_v5_own_commit_label_content is False
    assert (v4c.model.memory_v7_onset_ce_weight, v4b.model.memory_v7_onset_ce_weight) == (6.0, 3.0)
    assert (v4c.model.memory_v5_write_conf, v4c.model.memory_v7_write_debounce_steps, v4c.model.memory_v5_write_conf_min) == (0.3, 2, True)
    changed = {f.name for f in dataclasses.fields(v4b.model) if getattr(v4b.model, f.name) != getattr(v4c.model, f.name)}
    assert changed == {"memory_v5_own_commit_label_content", "memory_v7_onset_ce_weight"}
    for field in ("data", "weight_loader", "lr_schedule", "num_train_steps", "batch_size", "label_write_schedule_steps"):
        assert getattr(v4b, field) == getattr(v4c, field), field
    s = _config.get_config("pi05_yam_beans0922_v4c_smoke"); assert s.num_train_steps == 3 and not s.wandb_enabled


def test_v4d_is_v4c_without_the_two_tick_confirmation():
    """v4d (prepared 09-23 15:15 after the onset A/B): v4c minus the two-tick confirmation (under teacher forcing a one-tick
    label sentence can never be confirmed, so training banks lacked every "light on" note while rollout banks had them) plus
    the flip-back retraction within 2 ticks for deployment flicker. Everything else identical to v4c."""
    from openpi.training import config as _config

    v4c = _config.get_config("pi05_yam_beans0922_v4c")
    v4d = _config.get_config("pi05_yam_beans0922_v4d")
    assert (v4d.model.memory_v7_write_debounce_steps, v4c.model.memory_v7_write_debounce_steps) == (1, 2)
    assert (v4d.model.memory_v7_write_retract_steps, v4c.model.memory_v7_write_retract_steps) == (2, 0)
    assert v4d.model.memory_v5_own_commit_label_content is True and v4d.model.memory_v7_onset_ce_weight == 6.0
    changed = {f.name for f in dataclasses.fields(v4c.model) if getattr(v4c.model, f.name) != getattr(v4d.model, f.name)}
    assert changed == {"memory_v7_write_debounce_steps", "memory_v7_write_retract_steps"}, changed
    for field in ("data", "batch_size", "num_train_steps", "lr_schedule", "optimizer", "freeze_filter", "checkpoint_base_dir"):
        assert getattr(v4c, field) == getattr(v4d, field), field
    s = _config.get_config("pi05_yam_beans0922_v4d_smoke"); assert s.num_train_steps == 3 and not s.wandb_enabled


def test_v4e_moves_supervision_to_the_onsets_only():
    """v4e (09-23 16:30): v4c's write rule; only the window sampler (transition-anchored starts closer and more often) and the
    copy-tick weight of moving decision steps change."""
    from openpi.training import config as _config

    v4c = _config.get_config("pi05_yam_beans0922_v4c")
    v4e = _config.get_config("pi05_yam_beans0922_v4e")
    changed = {f.name for f in dataclasses.fields(v4c.model) if getattr(v4c.model, f.name) != getattr(v4e.model, f.name)}
    assert changed == {"memory_v6_decision_ce_weight_after_motion"}, changed
    assert (v4e.model.memory_v6_decision_ce_weight_after_motion, v4c.model.memory_v6_decision_ce_weight_after_motion) == (0.2, 1.0)
    dc, de = v4c.data.base_config, v4e.data.base_config
    data_changed = {f.name for f in dataclasses.fields(dc) if getattr(dc, f.name) != getattr(de, f.name)}
    assert data_changed == {"memory_critical_start_pad", "memory_critical_prob"}, data_changed
    assert (de.memory_critical_start_pad, de.memory_critical_prob) == (25, 0.7)
    assert (dc.memory_critical_start_pad, dc.memory_critical_prob) == (75, 0.5)
    assert v4e.model.memory_v5_own_commit_label_content is True and v4e.model.memory_v7_onset_ce_weight == 6.0
