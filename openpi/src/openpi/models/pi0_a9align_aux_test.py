"""A9 layer-8 auxiliary ablations: matched reader, causal writes, train/inference wiring."""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma, memory, pi0
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v5_test import _TinyV5Seq
from openpi.training import config


class TinyAlignedAux(_TinyV5Seq):
    _init_vis_layer8 = pi0.Pi0._init_vis_layer8
    vis_layer8_read_tokens = pi0.Pi0.vis_layer8_read_tokens
    vis_write_kv = pi0.Pi0.vis_write_kv
    vis_bank_write = pi0.Pi0.vis_bank_write
    vis_prefill = pi0.Pi0.vis_prefill
    v5_reference_template_rows = pi0.Pi0.v5_reference_template_rows
    _v5_whiten_map = pi0.Pi0._v5_whiten_map
    _v5_apply_whiten = staticmethod(pi0.Pi0._v5_apply_whiten)

    def __init__(self, *, image=True, state=False, oracle=True):
        rngs = nnx.Rngs(51)
        super().__init__(rngs, oracle_writes=oracle, pooling="standardized_attention")
        self.memory_v5_slot_keys = self.memory_v5_whiten_values = True
        self.memory_v5_slot_keep = pi0._v5_slot_templates(self.memory_v5_reference_tokens, 1)
        self.memory_v5_whiten_eps = 0.01
        self.memory_v5_query_standardize = self.memory_v5_query_prev_sentence = True
        self.memory_v5_prev_is_committed = not oracle
        self.memory_v0920_vision_outside_scan = not oracle
        self.memory_v4_visual_injection = False
        self.memory_mask_zero_tokens = True
        self.memory_sem_inst_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=rngs)
        self.memory_sem_inst_query_proj.kernel.value = jnp.eye(64)
        self.memory_sem_prev_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=rngs)
        self.memory_sem_prev_query_proj.kernel.value = jnp.zeros((64, 64))
        core = dataclasses.replace(self.memory.config, hidden_dims=(16, 16, 16))
        self.memory = memory.TitansMemory(core, rngs=rngs)
        self.memory_semantic = memory.TitansMemory(core, rngs=rngs)
        self.memory_vis_bank = self.memory_vis_layer8 = True
        self.memory_vis_image_write, self.memory_vis_state_slot = image, state
        self.memory_vis_slots = 3
        self.memory_vis_zero_read = False
        self.memory_vis_prefill_steps = 0
        self._init_vis_layer8(width=64, num_heads=8, dtype=jnp.float32, qk_norm=False, action_dim=2, rngs=rngs)


@pytest.fixture(autouse=True)
def small_vocab(monkeypatch):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)


def test_config_rejects_mixed_input_or_unconditioned_auxiliary_reads():
    model = config.get_config("pi05_yam_beans0922_ab_vis8_mlp3_a9align").model
    for changes in (
        dict(memory_v0920_input_read=True), dict(memory_v5_query_prev_sentence=False),
        dict(memory_v5_query_standardize=False), dict(memory_v7_no_visual_block=True),
        dict(memory_v4_visual_injection=True), dict(memory_vis_input_rms=1.0),
        dict(memory_vis_bank=False),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(model, **changes)


def test_auxiliary_conditioning_and_calibration_equal_a9_equations():
    m = TinyAlignedAux()
    # Parameter sharing is test-only: equality verifies the two equations, not
    # just matching output dimensions or the presence of a conditioner module.
    for tail in ("read_query_bank", "read_conditioner", "query_proj", "inst_query_proj", "prev_query_proj", "inject_w"):
        setattr(m, "memory_vis_" + tail, getattr(m, "memory_sem_" + tail))
    m.memory_vis_prev_query_proj.kernel.value = jnp.eye(64)
    context = jax.random.normal(jax.random.key(9), (1, 4, 64))
    mask = jnp.asarray([[True, True, False, False]])
    prev = jnp.asarray([[5, 6]])
    pmask = jnp.ones_like(prev, dtype=bool)
    bank, _ = m.memory.delta_write_kv(m.memory.init_state(1), jnp.ones((1, 1, 8)), jnp.ones((1, 1, 64)))
    read = m.vis_layer8_read_tokens(bank, context, mask, jnp.float32, prev_tokens=prev, prev_mask=pmask)
    expected_q = m.v5_semantic_queries(context, mask, prev, pmask)
    np.testing.assert_array_equal(read[3], expected_q)
    np.testing.assert_allclose(read[0], m._v4_inject_semantic(m.memory.read_key(bank, expected_q)), atol=1e-6)
    changed = m.vis_layer8_read_tokens(bank, context + 1, mask, jnp.float32, prev_tokens=prev, prev_mask=pmask)
    assert not np.allclose(read[3], changed[3])
    changed_prev = m.vis_layer8_read_tokens(bank, context, mask, jnp.float32,
                                          prev_tokens=jnp.asarray([[7, 8]]), prev_mask=pmask)
    assert not np.allclose(read[3], changed_prev[3])
    masked = m.vis_layer8_read_tokens(bank, context.at[:, 2:].set(999), mask, jnp.float32,
                                    prev_tokens=prev, prev_mask=pmask)
    np.testing.assert_array_equal(read[3], masked[3])


def test_auxiliary_initialization_matches_across_sources():
    models = [TinyAlignedAux(image=i, state=s) for i, s in ((True, False), (False, True), (True, True))]
    ref = nnx.state(models[0], nnx.Param)
    for m in models[1:]:
        jax.tree.map(np.testing.assert_array_equal, ref, nnx.state(m, nnx.Param))


def test_real_constructor_preserves_every_shared_a9_parameter(monkeypatch):
    # Use the REAL Pi0 constructor, with small transformer/image towers only.
    # This catches insertion of new RNG-consuming modules before common A9 leaves.
    from flax import linen as nn
    from openpi.models import siglip

    class Image(nn.Module):
        width: int

        @nn.compact
        def __call__(self, image, train=False):
            return nn.Dense(self.width)(jnp.mean(image, axis=(1, 2))[:, None]), None

    original = gemma.get_config
    def small_config(_):
        return dataclasses.replace(original("dummy"), depth=10)
    source = config.get_config("pi05_yam_beans0922_ab_snap_mlp3_a9align_A").model
    monkeypatch.setattr(gemma, "get_config", small_config)
    monkeypatch.setattr(siglip, "Module", lambda **kw: Image(kw["num_classes"]))
    bank = dataclasses.replace(source.memory, d_input=64, d_key=8, d_value=64, hidden_dims=(16, 16, 16))
    small = dataclasses.replace(source, memory=bank, memory_semantic=bank, memory_seq_steps=3,
                                max_token_len=4, causal_token_len=48, dtype="float32")
    base = pi0.Pi0(small, nnx.Rngs(42))
    auxiliary = pi0.Pi0(dataclasses.replace(small, memory_vis_bank=True, memory_vis_layer8=True), nnx.Rngs(42))
    common = dict(nnx.state(base, nnx.Param).flat_state())
    extended = dict(nnx.state(auxiliary, nnx.Param).flat_state())
    for path, value in common.items():
        np.testing.assert_array_equal(value.value, extended[path].value, err_msg=str(path))
    assert all(str(path[0]).startswith("memory_vis_") for path in extended.keys() - common.keys())


def test_layer8_prefix_is_memory_blind_and_writes_from_input_not_late_states():
    m = TinyAlignedAux(image=True, state=True)
    obs = _single_observation()
    prefix, mask, ar = m.embed_prefix(obs)
    keys, values, _ = m.vis_write_kv(prefix[:, :4], obs.state)
    blank = m.memory.init_state(1)
    written, _ = m.vis_bank_write(blank, keys, values, jnp.array([True]))
    sem = m.memory_semantic.init_state(1)
    def prepare(bank):
        return m._v32_prepare_memory_prefix(prefix, mask, ar, bank, top_token_count=4,
                                           semantic_state=sem, sensory_state=obs.state)
    first, full = prepare(blank), prepare(written)
    assert m._memory_token_total == 16 + 3 + 3
    np.testing.assert_array_equal(first["vis_retrieved"], 0)
    assert not np.any(first["vis_valid"])
    assert np.any(full["vis_valid"])
    np.testing.assert_array_equal(first["h8_all"], full["h8_all"])
    np.testing.assert_array_equal(first["sem_queries"], full["sem_queries"])
    np.testing.assert_array_equal(full["vis_keys"], keys)
    np.testing.assert_array_equal(full["vis_values"], values)
    np.testing.assert_array_equal(first["vis_keys"], full["vis_keys"])
    # No auxiliary columns in the early cache; content becomes visible only later.
    start = prefix.shape[1] + 16 + 3
    np.testing.assert_array_equal(full["cache"][0][:m.memory_layer + 1, :, start:start + 3], 0)
    assert np.any(np.asarray(full["cache"][0][m.memory_layer + 1:, :, start:start + 3]) != 0)
    m.memory_vis_zero_read = True
    zeroed = prepare(written)
    np.testing.assert_array_equal(zeroed["vis_retrieved"], 0)
    np.testing.assert_array_equal(zeroed["sem_retrieved"], full["sem_retrieved"])
    np.testing.assert_allclose(zeroed["final_prefix"], first["final_prefix"], atol=1e-6)


@pytest.mark.parametrize("image,state", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("oracle", [True, False])
def test_three_layer_slot_sequence_has_finite_losses_and_auxiliary_gradients(image, state, oracle):
    m = TinyAlignedAux(image=image, state=state, oracle=oracle)
    obs = _v4_sequence_observation()
    # Replay one masked padding row and one valid past row, with no history gradients.
    m.memory_vis_prefill_steps = 2
    obs = obs.replace(memory_vis_prefill_mask=jnp.asarray([[False, True]]),
                      memory_vis_prefill_state=obs.state[:, :2],
                      memory_vis_prefill_image=obs.images["base_0_rgb"][:, :2] if image else None)
    actions = jnp.zeros((1, 3, 4, 2))
    def objective(model):
        losses = model._compute_sequence_loss_v32(jax.random.key(13), obs, actions, train=False)
        return jnp.sum(losses["ce"]), losses
    (_, losses), grads = nnx.value_and_grad(objective, has_aux=True)(m)
    for name, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), name
    np.testing.assert_array_equal(losses["vis_commit_count"], 3)
    for path, value in jax.tree_util.tree_leaves_with_path(grads):
        assert np.isfinite(np.asarray(value)).all(), path
    for layer in range(3):
        assert np.any(np.asarray(grads["memory"]["m0"][f"w{layer}"].value) != 0), layer
    for name in ("memory_vis_key_proj", "memory_vis_value_proj", "memory_vis_query_proj",
                 *( ("memory_vis_pooler",) if image else () ),
                 *( ("memory_vis_state_proj",) if state else () )):
        assert any(np.any(np.asarray(v) != 0) for v in jax.tree.leaves(grads[name])), name


def test_prefill_detaches_history_but_not_static_hidden_weights():
    m = TinyAlignedAux(image=False, state=True)
    obs = _v4_sequence_observation().replace(
        memory_vis_prefill_mask=jnp.asarray([[False, True]]),
        memory_vis_prefill_state=jnp.asarray([[[99., 99.], [1., 2.]]]),
    )
    initial = m.memory.init_state(1)
    replayed = m.vis_prefill(obs, initial)
    keys, values, _ = m.vis_write_kv(jnp.zeros((1, 1, 64)), obs.memory_vis_prefill_state[:, 1])
    expected, _ = m.vis_bank_write(initial, keys, values, jnp.array([True]))
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6), replayed, expected)
    def output_history(model):
        return jnp.sum(model.vis_prefill(obs, model.memory.init_state(1)).fast_weights[model.memory._output_weight_name])
    history_grad = nnx.grad(output_history)(m)
    assert all(np.all(np.asarray(v) == 0) for v in jax.tree.leaves(history_grad))
    def hidden(model):
        return jnp.sum(model.vis_prefill(obs, model.memory.init_state(1)).fast_weights["w0"])
    hidden_grad = nnx.grad(hidden)(m)
    np.testing.assert_array_equal(hidden_grad["memory"]["m0"]["w0"].value, 1)


def test_layer8_serving_commits_the_same_auxiliary_write_and_frozen_preserves_state():
    m = TinyAlignedAux(image=False, state=True)
    obs = _single_observation()
    bank = m.memory.init_state(1)
    keys, values, _ = m.vis_write_kv(jnp.zeros((1, 4, 64)), obs.state)
    expected, _ = m.vis_bank_write(bank, keys, values, jnp.array([True]))
    kwargs = dict(stop_token=1, max_decode_steps=1, num_steps=1, noise=jnp.zeros((1, 4, 2)),
                  semantic_state=m.memory_semantic.init_state(1), v35_transition_valid=True, v35_write_mask=True,
                  forced_subtask_tokens=jnp.asarray([[5, 6]]), forced_subtask_mask=jnp.ones((1, 2), dtype=bool))
    _, written, _ = m.sample_with_memory(jax.random.key(4), obs, bank, write_mode="normal", **kwargs)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6), written, expected)
    _, frozen, _ = m.sample_with_memory(jax.random.key(4), obs, bank, write_mode="frozen", **kwargs)
    jax.tree.map(np.testing.assert_array_equal, frozen, bank)
