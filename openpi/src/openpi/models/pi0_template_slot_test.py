"""Template discovery, matched-address reads, occupancy and sequence gradients."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma, memory, pi0
from openpi.models.pi0_v0920_test import _TinyV0
from openpi.models.pi0_v0922ab_test import _attach_vis
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.sentence_slots import template_masks, template_representatives


class TinySlot(_TinyV0):
    v5_reference_template_rows = pi0.Pi0.v5_reference_template_rows
    v5_template_keys = pi0.Pi0.v5_template_keys
    v5_template_read_tokens = pi0.Pi0.v5_template_read_tokens
    _vis_input_scale = pi0.Pi0._vis_input_scale
    vis_write_kv = pi0.Pi0.vis_write_kv
    vis_bank_write = pi0.Pi0.vis_bank_write
    vis_read_tokens = pi0.Pi0.vis_read_tokens
    vis_prefill = pi0.Pi0.vis_prefill

    def __init__(self, hidden_dims=()):
        super().__init__(nnx.Rngs(9), history_frames=0)
        self.memory_template_read = True
        self.memory_v6_token_writes = False
        self.memory_v5_slot_keys = True
        self.memory_v5_whiten_values = True
        self.memory_v5_slot_max_diff = 1
        self.memory_v5_reference_tokens = ((5, 6), (5, 7), (8, 9), (10, 11), (12, 13), (14, 15))
        self.memory_v5_slot_keep = template_masks(self.memory_v5_reference_tokens, 1)
        self.memory_template_rows = template_representatives(self.memory_v5_reference_tokens, 1)
        self.memory_v5_read_queries = len(self.memory_template_rows)
        self.memory_sem_slot_embedding = nnx.Param(jnp.full((self.memory_v5_read_queries, 64), 0.1))
        self.memory_semantic = memory.TitansMemory(dataclasses.replace(
            self.memory_semantic.config, hidden_dims=hidden_dims, slot_count=self.memory_v5_read_queries
        ), rngs=nnx.Rngs(11))


@pytest.fixture(params=[(), (32, 32, 32)], ids=["linear", "mlp3"])
def model(request):
    old = gemma.PALIGEMMA_VOCAB_SIZE
    gemma.PALIGEMMA_VOCAB_SIZE = 128
    try:
        yield TinySlot(hidden_dims=request.param)
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = old


def test_discovery_is_vocabulary_derived():
    rows = ((1, 2, 3), (1, 2, 4), (8,), (9, 10))
    assert template_masks(rows, 1) == ((True, True, False), (True, True, False), (True,), (True, True))
    assert template_representatives(rows, 1) == (0, 2, 3)
    assert template_representatives(rows + ((10, 11, 12, 13),), 1) == (0, 2, 3, 4)


def test_read_matches_write_addresses_and_masks_unwritten_slots(model):
    blank = model.memory_semantic.init_state(1)
    assert not np.asarray(model.v0920_read_tokens(blank, 1, jnp.float32)[1]).any()
    tokens = jnp.array([[5, 6]], dtype=jnp.int32)
    mask = jnp.ones_like(tokens, dtype=bool)
    written, applied = model.v5_commit_sentence(blank, tokens, mask, jnp.array([True]))
    assert bool(applied[0])
    out, valid, retrieved, queries, *_ = model.v0920_read_tokens(written, 1, jnp.float32)
    np.testing.assert_array_equal(valid, [[True, False, False, False, False]])
    np.testing.assert_array_equal(out[:, 1:], 0)
    keys = model.v5_sentence_kv(tokens, mask)[0]
    np.testing.assert_allclose(keys[:, 0], queries[:, 0], atol=1e-6)
    np.testing.assert_allclose(retrieved[:, :1], model.memory_semantic.read_key(written, keys), atol=1e-6)
    decayed, _ = model.memory_semantic.analytic_decay(written, jnp.array([10]))
    np.testing.assert_array_equal(decayed.slot_written, written.slot_written)
    zero = model.v0920_read_tokens(written, 1, jnp.float32, zero_read=True)
    np.testing.assert_array_equal(zero[0], 0)
    assert not np.asarray(zero[1]).any()
    rewritten, _ = model.v5_commit_sentence(written, jnp.array([[5, 7]]), mask, jnp.array([True]))
    np.testing.assert_array_equal(rewritten.slot_written, written.slot_written)
    unknown, applied = model.v5_commit_sentence(rewritten, jnp.array([[70, 71]]), mask, jnp.array([True]))
    assert not bool(applied[0])
    np.testing.assert_array_equal(unknown.slot_written, written.slot_written)


@pytest.mark.parametrize("oracle", [True, False], ids=["A", "B"])
def test_sequence_loss_and_gradients_are_finite(model, oracle):
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    model.memory_v5_oracle_writes = oracle
    model.memory_v7_write_every_step = False
    model.memory_v5_write_conf = 0.9
    model.memory_v5_write_conf_min = False
    def loss(m):
        losses = m._compute_sequence_loss_v32(jax.random.key(8), observation, actions, train=False)
        return jnp.mean(losses["ce"] + losses["flow"])
    value, grads = nnx.value_and_grad(loss)(model)
    assert np.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))


def test_snap_mlp3_uses_mean_confidence_over_valid_sentence_tokens():
    from openpi.training import config

    settings = config.get_config("pi05_yam_beans0922_ab_snap_mlp3").model
    probabilities = jnp.array([[0.95, 0.95, 0.0], [0.89, 0.89, 0.0], [0.99, 0.99, 0.75], [0.9, 0.0, 0.0]])
    spans = jnp.array([[True, True, False], [True, True, False], [True, True, True], [True, False, False]])
    confidence = pi0.v5_sentence_confidence(settings, probabilities, spans)
    np.testing.assert_allclose(confidence, [0.95, 0.89, 0.91, 0.9], atol=1e-6)
    np.testing.assert_array_equal(confidence >= settings.memory_v5_write_conf, [True, False, True, True])
    np.testing.assert_array_equal(pi0.v5_sentence_confidence(settings, probabilities, jnp.zeros_like(spans)), 0)


def test_production_mlp3_delta_changes_only_the_output_matrix():
    from openpi.training import config

    settings = config.get_config("pi05_yam_beans0922_ab_snap_mlp3").model.memory_semantic
    bank = memory.TitansMemory(settings, rngs=nnx.Rngs(41))
    blank = bank.init_state(1)
    assert bank._output_weight_name == "w3"
    assert blank.fast_weights["w0"].shape == (1, 512, 1024)
    assert blank.fast_weights["w1"].shape == blank.fast_weights["w2"].shape == (1, 1024, 1024)
    assert blank.fast_weights["w3"].shape == (1, 1024, 2048)
    keys = jax.random.normal(jax.random.key(42), (1, 1, settings.d_key))
    values = jax.random.normal(jax.random.key(43), (1, 1, settings.d_value))
    state = blank
    # Replacing a value at the same address tests a residual delta write, not additive accumulation.
    for value in (values, -values):
        state, aux = bank.delta_write_kv_multi(state, keys, value, jnp.array([[True]]))
        assert bool(aux["commit_applied"][0, 0])
        assert np.linalg.norm(np.asarray(state.fast_weights["w3"])) > 0
        np.testing.assert_allclose(bank.read_key(state, keys), aux["pooled_value"], atol=2e-6)
        for name, initial in blank.fast_weights.items():
            if name != "w3":
                np.testing.assert_array_equal(state.fast_weights[name], initial)
        for momentum in state.momentum.values():
            np.testing.assert_array_equal(momentum, 0)
    decayed, _ = bank.analytic_decay(state, jnp.array([1]))
    np.testing.assert_allclose(decayed.fast_weights["w3"], state.fast_weights["w3"] * 0.999, atol=1e-7)
    for name in ("w0", "w1", "w2"):
        np.testing.assert_array_equal(decayed.fast_weights[name], blank.fast_weights[name])


@pytest.mark.parametrize('image,state', [(True, False), (True, True), (False, True)])
def test_template_slots_and_past_auxiliary_replay_train_together(model, image, state):
    _attach_vis(model, nnx.Rngs(18), on=True, image=image, state=state, rule='delta')
    model.memory_vis_prefill_steps = 2
    model.memory_v5_oracle_writes = True
    observation = _v4_sequence_observation().replace(
        memory_vis_prefill_mask=jnp.array([[False, True]]),
        memory_vis_prefill_state=jnp.ones((1, 2, 2)),
        memory_vis_prefill_image=jnp.ones((1, 2, 224, 224, 3)) if image else None,
    )
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)

    def loss(m):
        losses = m._compute_sequence_loss_v32(jax.random.key(23), observation, actions, train=False)
        return jnp.mean(losses['ce'] + losses['flow'])

    value, grads = nnx.value_and_grad(loss)(model)
    assert np.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads['memory_vis_value_proj']))
