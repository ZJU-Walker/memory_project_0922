"""beans0922 ablation configs: the ablation rows differ from snap (pi05_yam_beans0922_v4b since 09-23 05:30; v4, v3, v1 before) in the
recipe fields only, and the visual-bank row adds exactly the visual bank; the new parameters fall under the fresh-init /
memory-leaf rules."""

import dataclasses
import re

from openpi.training import config as _config  # first: config.py imports the beans0922 modules itself (circular otherwise)
from openpi.training import beans0922_ablation_config as ab  # noqa: E402, I001
from openpi.training import beans0922_config as b  # noqa: E402, I001

RECIPE_FIELDS = {"name", "fsdp_devices", "num_workers", "project_name", "save_interval", "keep_period", "num_train_steps",
                 "label_write_schedule_steps", "batch_size"}


def _diff(a, b_):
    return {f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b_, f.name)}


def test_beans0922_ablation_configs():
    snap = _config.get_config("pi05_yam_beans0922_v4b")
    ab_snap = _config.get_config("pi05_yam_beans0922_ab_snap")
    vis8 = _config.get_config("pi05_yam_beans0922_ab_vis8")
    # snap itself is untouched by the ablation flags
    assert snap.model.memory_vis_bank is False and snap.model.memory_vis_zero_read is False
    # the control row = snap + the recipe
    assert _diff(ab_snap, snap) <= RECIPE_FIELDS, _diff(ab_snap, snap)
    assert ab_snap.model == snap.model
    assert ab_snap.num_train_steps == ab.AB_STEPS + 1 and ab_snap.label_write_schedule_steps == 500
    assert ab_snap.fsdp_devices == ab.AB_FSDP and ab_snap.project_name == ab.PROJECT
    assert ab_snap.checkpoint_base_dir == snap.checkpoint_base_dir  # own dir per config name, no collision
    # the visual-bank row = the control row + the visual bank (+ its gate frozen like the sentence gate)
    assert _diff(vis8, ab_snap) == {"name", "model", "freeze_filter"}
    for leaf, frozen in (("memory_vis_inject_w", True), ("memory_sem_inject_w", True), ("memory_vis_pooler/query_bank", False),
                         ("memory_vis_key_proj/kernel", False), ("memory_vis_read_query_bank", False), ("memory_vis_slot_embedding", False),
                         ("memory_sem_read_query_bank", False), ("PaliGemma/img/Transformer/x", True), ("PaliGemma/llm/layers/attn/q", False)):
        assert bool(vis8.freeze_filter.pattern.match(leaf)) is frozen, leaf
        if leaf != "memory_vis_inject_w":
            assert bool(ab_snap.freeze_filter.pattern.match(leaf)) is frozen, leaf
    assert not ab_snap.freeze_filter.pattern.match("memory_vis_inject_w")  # snap's own filter is untouched
    assert _diff(vis8.model, ab_snap.model) == {"memory", "memory_vis_bank"}  # 8 slots = the field default
    assert vis8.model.memory_vis_bank and vis8.model.memory_vis_slots == 8
    assert vis8.model.memory == vis8.model.memory_semantic  # the same linear delta-rule bank as the sentence bank
    assert vis8.model.memory.hidden_dims == () and vis8.model.memory.write_rule == "delta_output"
    # the sensory rows: image / state slots and the commit rule are the only differences
    expected = {
        "vis8s": ({"memory", "memory_vis_bank", "memory_vis_state_slot"}, True, True, "delta"),
        "vis8s_add": ({"memory", "memory_vis_bank", "memory_vis_state_slot"}, True, True, "additive"),
        "state8": ({"memory", "memory_vis_bank", "memory_vis_state_slot", "memory_vis_image_write"}, False, True, "delta"),
        "state8_add": ({"memory", "memory_vis_bank", "memory_vis_state_slot", "memory_vis_image_write"}, False, True, "additive"),
    }
    assert set(ab.ROWS) == {"vis8", *expected}
    for suffix, (fields, image, state, rule) in expected.items():
        row = _config.get_config(f"pi05_yam_beans0922_ab_{suffix}")
        assert _diff(row, ab_snap) == {"name", "model", "freeze_filter"}, suffix
        assert row.freeze_filter == vis8.freeze_filter, suffix
        assert _diff(row.model, ab_snap.model) == fields, (suffix, _diff(row.model, ab_snap.model))
        assert row.model.memory_vis_image_write is image and row.model.memory_vis_state_slot is state, suffix
        assert row.model.memory.commit_rule == rule and row.model.memory_semantic.commit_rule == "delta", suffix
        assert dataclasses.replace(row.model.memory, commit_rule="delta") == row.model.memory_semantic, suffix
        smoke = _config.get_config(f"pi05_yam_beans0922_ab_{suffix}_smoke")
        assert smoke.model == row.model and smoke.num_train_steps == 3 and smoke.wandb_enabled is False, suffix
    assert vis8.model.memory.blank_initial_output and vis8.model.memory.alpha_step == vis8.model.memory_semantic.alpha_step  # one decay for both banks (v4: 0.001)
    assert vis8.model.memory_v0920_input_read and vis8.model.memory_v35_enabled
    # the new leaves are memory leaves (fresh init from the base, memory-group grad clip)
    for leaf in ("memory_vis_pooler/query_bank", "memory_vis_key_proj/kernel", "memory_vis_value_proj/kernel",
                 "memory_vis_slot_key", "memory_vis_read_query_bank", "memory_vis_query_proj/kernel",
                 "memory_vis_inject_w", "memory_vis_slot_embedding"):
        assert re.fullmatch(_config._V7_MEMORY_LEAF, leaf), leaf  # noqa: SLF001
        assert not re.fullmatch(_config._V7_NON_MEMORY_LEAF, leaf), leaf  # noqa: SLF001
    assert vis8.weight_loader.fresh_init_allowlist == (_config._V7_MEMORY_LEAF,)  # noqa: SLF001
    assert vis8.weight_loader.params_path == b.base_params_path()
    # smokes: 2 updates, no W&B, same models
    for name in ("pi05_yam_beans0922_ab_snap_smoke", "pi05_yam_beans0922_ab_vis8_smoke"):
        smoke = _config.get_config(name)
        assert smoke.num_train_steps == 3 and smoke.wandb_enabled is False
    assert _config.get_config("pi05_yam_beans0922_ab_vis8_smoke").model == vis8.model


def test_every_row_is_built_on_the_v4b_snap():
    """User 09-23 05:30 (relayed by the base session): all ablation rows adopt v4b = v4 (pointer bonus, last-note read-back of
    48 tokens, decay 0.999/tick on both banks, wrong-token weight 5, no v3 question shift) with the write gate the model's own
    notes can pass: every word >= 0.3 and the same sentence on two consecutive ticks (v4's 0.8 gate wrote one note per episode)."""
    v1 = _config.get_config("pi05_yam_beans0922_v1").model
    v4 = _config.get_config("pi05_yam_beans0922_v4b").model
    assert ab.SNAP_OVERRIDES == b.V4B_WRITE_RULE and ab.SNAP_BANK_OVERRIDES == b.V4_BANK
    assert v4.memory_v5_write_conf == 0.3 and v4.memory_v7_write_debounce_steps == 2 and v4.memory_v5_write_conf_min
    assert v1.memory_v7_write_every_step and v1.memory_v5_write_conf == 0.0 and not v1.memory_v0920_prev_readback
    assert not v4.memory_v0920_query_context  # v3's shift is gone
    for name in ["pi05_yam_beans0922_ab_snap"] + [f"pi05_yam_beans0922_ab_{row}" for row in ab.ROWS]:
        model = _config.get_config(name).model
        for key, value in b.V4B_WRITE_RULE.items():
            assert getattr(model, key) == value, (name, key)
        assert not model.memory_v0920_query_context
        assert float(model.memory_semantic.alpha_step) == float(model.memory.alpha_step) == 0.001, name  # both banks, one decay
        # the sensory rows differ from the v4 snap in the bank fields only
        assert {f.name for f in dataclasses.fields(model) if getattr(model, f.name) != getattr(v4, f.name)} <= {
            "memory", "memory_vis_bank", "memory_vis_image_write", "memory_vis_state_slot"}, name
        if model.memory_vis_bank:  # the sensory bank = the sentence bank's config + the row's commit rule
            assert dataclasses.replace(model.memory, commit_rule="delta") == model.memory_semantic
