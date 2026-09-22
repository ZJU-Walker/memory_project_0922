"""beans0922 ablation configs: the ablation rows differ from snap (pi05_yam_beans0922_v1) in the recipe fields only, and
the visual-bank row adds exactly the visual bank; the new parameters fall under the fresh-init / memory-leaf rules."""

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
    snap = _config.get_config("pi05_yam_beans0922_v1")
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
    # the visual-bank row = the control row + the visual bank
    assert _diff(vis8, ab_snap) == {"name", "model"}
    assert _diff(vis8.model, ab_snap.model) == {"memory", "memory_vis_bank"}  # 8 slots = the field default
    assert vis8.model.memory_vis_bank and vis8.model.memory_vis_slots == 8
    assert vis8.model.memory == vis8.model.memory_semantic  # the same linear delta-rule bank as the sentence bank
    assert vis8.model.memory.hidden_dims == () and vis8.model.memory.write_rule == "delta_output"
    assert vis8.model.memory.blank_initial_output and vis8.model.memory.alpha_step == 0.01
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
