"""Template-slot A/B ablation recipe: shared controls and row-specific auxiliary bank."""
import dataclasses

import pytest

from openpi.training import config
from openpi.training import beans0922_ablation_config as ab
from openpi.models.sentence_slots import template_representatives


@pytest.mark.parametrize("row", ab.ALL_ROWS)
def test_rows_have_matched_own_a_b_stages(row):
    a = config.get_config(f"pi05_yam_beans0922_ab_{row}_A")
    b = config.get_config(f"pi05_yam_beans0922_ab_{row}")
    assert a.num_train_steps == ab.A_STEPS + 1
    assert b.num_train_steps == ab.AB_STEPS + 1
    assert a.model.memory_v5_oracle_writes and not b.model.memory_v5_oracle_writes
    assert dataclasses.replace(a.model, memory_v5_oracle_writes=False) == b.model
    assert a.batch_size == b.batch_size == ab.AB_BATCH
    assert a.gradient_accumulation_steps == b.gradient_accumulation_steps == ab.AB_ACCUM
    assert a.seed == b.seed == 42
    assert a.label_write_schedule_steps == b.label_write_schedule_steps == 0
    assert a.fsdp_devices == b.fsdp_devices == ab.AB_FSDP
    assert a.lr_schedule == b.lr_schedule
    assert a.data == b.data
    assert a.weight_loader.params_path.endswith("beans0922_base/10000/params")
    assert b.weight_loader.params_path == ab.stage_a_params(row)
    assert b.weight_loader.matched_allowlist == (r".+",)
    assert b.weight_loader.fresh_init_allowlist == ()
    for stage in ("", "_A"):
        smoke = config.get_config(f"pi05_yam_beans0922_ab_{row}{stage}_smoke")
        assert smoke.num_train_steps == 3 and not smoke.wandb_enabled


@pytest.mark.parametrize("row", ab.ALL_ROWS)
def test_all_rows_use_whole_sentence_direct_template_reads(row):
    model = config.get_config(f"pi05_yam_beans0922_ab_{row}").model
    assert model.memory_template_read and model.memory_v0920_input_read
    assert model.memory_v5_slot_keys and model.memory_v5_whiten_values
    assert not model.memory_v6_token_writes
    assert not model.memory_v6_pointer_read
    assert not model.memory_v0920_prev_readback
    assert not model.memory_v0920_query_context
    assert not model.memory_v5_own_commit_label_content
    assert model.memory_v7_write_debounce_steps == 1
    assert model.memory_v5_write_conf == (0.9 if row == "snap_mlp3" else 0.3)
    assert model.memory_v5_write_conf_min == (row != "snap_mlp3")
    count = len(template_representatives(model.memory_v5_reference_tokens))
    assert count == model.memory_v5_read_queries == model.memory_semantic.slot_count == 5
    assert model.memory_semantic.alpha_step == model.memory.alpha_step == 0.001
    assert model.memory.slot_count == 0
    assert model.memory_vis_bank == (row in ab.ROWS)
    if row in ab.ROWS:
        image, state, rule = ab.ROWS[row]
        assert model.memory_vis_image_write == image
        assert model.memory_vis_state_slot == state
        assert model.memory.commit_rule == rule
        assert model.memory_vis_prefill_steps == ab.PREFILL_STEPS
        assert model.memory_vis_slots == 8
        assert dataclasses.replace(model.memory, commit_rule="delta", slot_count=count) == model.memory_semantic
    else:
        assert model.memory_vis_prefill_steps == 0


def test_only_auxiliary_fields_differ_from_snap():
    snap = config.get_config("pi05_yam_beans0922_ab_snap")
    for row in ab.ROWS:
        variant = config.get_config(f"pi05_yam_beans0922_ab_{row}")
        fields = {f.name for f in dataclasses.fields(snap.model) if getattr(snap.model, f.name) != getattr(variant.model, f.name)}
        assert fields <= {"memory", "memory_vis_bank", "memory_vis_image_write", "memory_vis_state_slot", "memory_vis_prefill_steps"}
        assert variant.data == snap.data
        assert variant.batch_size == snap.batch_size
        assert variant.seed == snap.seed


def test_snap_mlp3_changes_only_the_requested_controls():
    snap = config.get_config("pi05_yam_beans0922_ab_snap")
    variant = config.get_config("pi05_yam_beans0922_ab_snap_mlp3")
    bank = variant.model.memory_semantic
    assert bank.hidden_dims == (1024, 1024, 1024)
    assert bank.dims == (512, 1024, 1024, 1024, 2048)
    assert bank.write_rule == "delta_output" and bank.commit_rule == "delta"
    assert bank.delta_rate == 1.0 and bank.alpha_step == 0.001
    assert dataclasses.replace(bank, hidden_dims=()) == snap.model.memory_semantic
    assert dataclasses.replace(
        variant.model, memory_semantic=snap.model.memory_semantic,
        memory_v5_write_conf=0.3, memory_v5_write_conf_min=True,
    ) == snap.model
    assert variant.model.simulated_delay == 15
    assert variant.model.memory_seq_steps == 40 and variant.model.memory_block_steps == 25
    assert variant.model.memory_state_mask_prob == 0
    d = variant.data.base_config
    assert d.memory_stride_frames == 5
    assert d.memory_critical_start_pad == 50
    assert d.memory_critical_prob == 0.5 and d.memory_slice_prob == 0.5
    assert (1 - d.memory_critical_prob) * (1 - d.memory_slice_prob) == 0.25
    assert (1 - d.memory_critical_prob) * d.memory_slice_prob == 0.25
    assert dataclasses.replace(
        variant.data, base_config=dataclasses.replace(d, memory_critical_prob=0.7, memory_critical_start_pad=25)
    ) == snap.data
    assert variant.lr_schedule == snap.lr_schedule
    assert variant.optimizer == snap.optimizer
    assert variant.freeze_filter == snap.freeze_filter
    assert variant.batch_size == snap.batch_size
    assert variant.memory_grad_clip == snap.memory_grad_clip
