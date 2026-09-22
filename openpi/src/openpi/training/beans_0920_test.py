"""pi05_yam_beans_0920_v1 = the 0920 v1 memory structure on the bean-scoop recipe: structure flags equal the RoboMME v1 run,
window / tick / labels / base / optimiser equal boba2B (+ boba3B data), three YAM cameras, own writes with the label ramp."""
import dataclasses
import pathlib

import pytest


def test_beans_v1_is_the_v1_structure_on_the_boba_recipe():
    from openpi.training import config as _config
    from openpi.training import beans_0920_config as b

    root = pathlib.Path(__file__).resolve().parents[4]
    if not (root / "robomme/metadata/PickXtimes/prepared_official.json").is_file():
        pytest.skip("no PickXtimes spec (the v1 reference config)")
    beans, v1, boba = (_config.get_config(n) for n in ("pi05_yam_beans_0920_v1", "pi05_robomme_0920_v1", "pi05_yam_mem_v7_boba2B"))
    # structure = v1
    for k in b.STRUCTURE:
        if k != "memory_v0920_drop_blank_camera":  # a camera choice: RoboMME has a blank slot to drop, the YAM station has three real cameras
            assert getattr(beans.model, k) == getattr(v1.model, k), k
    assert (beans.model.memory_v5_oracle_writes, beans.model.memory_v5_prev_is_committed, beans.model.memory_v5_own_commit_label_content) == (False, True, False)
    assert beans.model.memory_v0920_input_read and not beans.model.memory_v6_pointer_read and beans.model.memory_v7_write_every_step
    assert beans.model.memory_v0920_history_frames == 0 and not beans.model.memory_v0920_drop_blank_camera
    # everything else = the bean scoop
    for k in ("action_horizon", "max_token_len", "memory_seq_steps", "memory_block_steps", "simulated_delay", "memory_v5_prefill_max",
              "memory_state_mask_prob", "memory_v5_reference_tokens", "memory_semantic"):
        assert getattr(beans.model, k) == getattr(boba.model, k), k
    assert (beans.model.action_horizon, beans.model.memory_seq_steps, beans.model.simulated_delay) == (50, 60, 15)
    d, db = beans.data.base_config, boba.data.base_config
    for k in ("memory_stride_frames", "memory_min_slice_steps", "memory_sequence_buckets", "memory_slice_prob", "memory_critical_prob",
              "memory_critical_start_pad", "memory_start_history_power", "memory_subtask_vocab", "evidence_subtasks", "memory_required_subtasks",
              "memory_waiting_state_dim", "memory_v5_generic_task", "memory_manifest_split"):
        assert getattr(d, k) == getattr(db, k), k
    assert (d.memory_stride_frames, d.memory_sequence_buckets, len(d.memory_subtask_vocab)) == (15, (20, 40, 60), 25)
    assert d.memory_v5_subtask_labels_path.endswith(b.BOBA3_SIDECAR) and d.memory_v5_subtask_labels_sha256 == b.BOBA3_SIDECAR_SHA256
    assert d.memory_episode_manifest_path.endswith(b.BOBA3_MANIFEST) and d.memory_episode_manifest_sha256 == b.BOBA3_MANIFEST_SHA256
    assert d.lerobot_dataset_root.endswith("boba_0913_v2") and beans.data.repo_id == b.BOBA3_REPO_ID and d.memory_image_history_frames == 0
    assert beans.data.assets == boba.data.assets
    # base weights, optimiser, schedule, launch shape
    assert beans.weight_loader.params_path.endswith(b.BOBA_BASE) and beans.weight_loader.params_path == boba.weight_loader.params_path
    assert beans.weight_loader.matched_allowlist == boba.weight_loader.matched_allowlist and beans.weight_loader.fresh_init_allowlist == boba.weight_loader.fresh_init_allowlist
    assert beans.lr_schedule == boba.lr_schedule and beans.optimizer == boba.optimizer and beans.memory_grad_clip == boba.memory_grad_clip
    assert beans.freeze_filter == boba.freeze_filter
    assert (beans.batch_size, beans.fsdp_devices, beans.num_train_steps, beans.label_write_schedule_steps) == (b.BEANS_BATCH, 2, 3001, 900)
    assert beans.checkpoint_base_dir.endswith("beans/checkpoints") and beans.project_name == "beans_0920"
    spec_obs, spec_act = beans.model.inputs_spec(batch_size=1)
    assert sorted(spec_obs.images) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"] and spec_act.shape[-2:] == (50, 32)
    smoke = _config.get_config("pi05_yam_beans_0920_v1_smoke")
    assert smoke.num_train_steps == 3 and not smoke.wandb_enabled and smoke.model == beans.model
