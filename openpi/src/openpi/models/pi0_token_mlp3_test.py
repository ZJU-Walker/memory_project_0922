"""Token writes into a three-hidden-layer bank, with the unchanged conditioned reader.

No pointer or input read-back. CPU tests use the real writer/sequence implementation
at tiny transformer width; the final test also exercises the full 512/1024/2048 bank.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma, memory
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v5_test import _with_prefill
from openpi.models.pi0_v6_test import _TinyV6Seq, _rows


@pytest.fixture
def model():
    old = gemma.PALIGEMMA_VOCAB_SIZE
    gemma.PALIGEMMA_VOCAB_SIZE = 128
    try:
        m = _TinyV6Seq(nnx.Rngs(7), pointer_read=False, hidden_dims=(64, 64, 64),
                       sentence_len=2, reference_tokens=((5, 6), (7, 8), (5,)), pooling="standardized_attention")
        m.memory_v6_whiten_keys = True
        m.memory_v4_visual_injection = False
        m.memory_v5_query_standardize = True
        m.memory_v5_query_prev_sentence = True
        m.memory_sem_inst_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=nnx.Rngs(14))
        m.memory_sem_inst_query_proj.kernel.value = jnp.eye(64, dtype=jnp.float32)
        m.memory_sem_prev_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=nnx.Rngs(13))
        m.memory_sem_prev_query_proj.kernel.value = jnp.zeros((64, 64), dtype=jnp.float32)
        yield m
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = old


def test_token_mlp3_masks_padding_and_changes_only_output_matrix(model):
    tokens, mask = _rows((5, 6), (7,), length=2)
    bank = model.memory_semantic
    blank = bank.init_state(2)
    written, aux, _ = model.v6_semantic_write_tokens(blank, tokens, mask, jnp.asarray([True, True]))
    np.testing.assert_array_equal(aux["commit_applied"], mask)
    assert not hasattr(model, "memory_v6_pointer_query_proj")
    assert model._memory_token_total == 16 + model.memory_v5_read_queries
    for name, before in blank.fast_weights.items():
        if name != bank._output_weight_name:
            np.testing.assert_array_equal(written.fast_weights[name], before)
    for leaf in written.momentum.values():
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))
    decayed, aux, _ = model.v6_semantic_write_tokens(written, tokens, mask, jnp.asarray([False, False]))
    expected, _ = bank.analytic_decay(written, 1)
    assert not np.asarray(aux["commit_applied"]).any()
    for name in expected.fast_weights:
        np.testing.assert_array_equal(decayed.fast_weights[name], expected.fast_weights[name])


@pytest.mark.parametrize("oracle", [True, False])
def test_conditioned_token_mlp3_sequence_prefill_has_finite_losses_and_gradients(model, oracle):
    model.memory_v5_oracle_writes = oracle
    model.memory_v5_prev_is_committed = not oracle
    model.memory_v5_write_delay_steps = 0
    model.memory_v5_write_conf = 0.9
    model.memory_v5_write_conf_min = False
    model.memory_v0920_vision_outside_scan = not oracle
    model.memory_v5_prefill_history = True
    model.memory_v5_prefill_max = 2
    obs = _with_prefill(_v4_sequence_observation(), sentences=[[5, 6]], gaps=[2], pending=[7, 8])
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(46), obs, actions, train=False)
    for key, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), key
    np.testing.assert_array_equal(losses["v5_prefill_sentence_count"], 1.0)

    def loss(m):
        return jnp.sum(m._compute_sequence_loss_v32(jax.random.key(46), obs, actions, train=False)["v4_decision_ce_steps"])

    grads = nnx.grad(loss)(model)
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        assert np.isfinite(np.asarray(leaf)).all(), path
    if oracle:
        for name in ("memory_v6_token_key_proj", "memory_v6_token_value_proj"):
            assert float(jnp.max(jnp.abs(grads[name]["kernel"].value))) > 0


def test_production_width_token_scan_is_output_only_and_decays_once():
    bank = memory.TitansMemory(memory.MemoryConfig(
        d_input=2048, d_key=512, hidden_dims=(1024, 1024, 1024), d_value=2048,
        mlp_l2norm=True, blank_initial_output=True, write_rule="delta_output", association_mode="pooled_frame",
        delta_rate=1.0, alpha_step=0.01,
    ), rngs=nnx.Rngs(42))
    state = bank.init_state(1)
    keys = memory.l2_normalize(jax.random.normal(jax.random.key(11), (1, 4, 512)))
    values = memory.l2_normalize(jax.random.normal(jax.random.key(12), (1, 4, 2048)))
    # Seed a nonzero bank so repeated decay would be observable.
    state, _ = bank.delta_write_kv(state, keys[:, :1], values[:, :1])
    mask = jnp.asarray([[True, True, False, True]])
    written, aux = bank.delta_write_kv_multi(state, keys, values, mask, slot_loop="scan")
    for name in state.fast_weights:
        if name != bank._output_weight_name:
            np.testing.assert_array_equal(written.fast_weights[name], state.fast_weights[name])
    hidden = np.asarray(bank.hidden_key(state, keys))[0]
    expected = np.asarray(state.fast_weights[bank._output_weight_name])[0] * 0.99
    for i in (0, 1, 3):
        h = hidden[i]
        residual = np.asarray(values)[0, i] - h @ expected
        expected = expected + np.outer(h, residual) / (h @ h)
    np.testing.assert_allclose(written.fast_weights[bank._output_weight_name][0], expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(aux["commit_applied"], mask)
    reads = np.asarray(bank.read_key(written, keys))[0]
    np.testing.assert_allclose(reads[3], np.asarray(values)[0, 3], atol=3e-6)
    assert np.isfinite(reads).all()  # earlier tokens may have interference; not an accuracy assertion
