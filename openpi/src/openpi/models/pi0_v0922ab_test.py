"""beans0922 ablation (1), Pi0Config.memory_vis_bank: a visual bank next to the sentence bank, read at the input.

Adversarial checks on the tiny 0920 stand-in: (a) the config gate; (b) with the flag OFF the model is snap: same tokens,
same losses as the original tiny_v0 class; (c) an empty visual bank reads exactly zero, masked, so tick 0 is bit-identical
to snap; a written bank reads finite tokens at the image-token scale; the zero-read switch; (d) the write content is
memory-blind and stop-gradient (no gradient reaches anything but the memory_vis_* leaves); (e) the transition: commit on
every valid tick, exact decay when not committing, invalid ticks keep the state, a second association does not destroy
the first; (f) the sequence loss is finite, commits on every valid tick, its gradient reaches every visual leaf, and with
the visual read zeroed the trained losses equal snap's exactly (the write path cannot leak into the loss); (g) the
sampler advances the visual bank in "normal" mode, keeps it in "frozen", decays it in "dynamics_only".
"""

# ruff: noqa: SLF001

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v0920_test import _TinyV0, _single_step_observation, _v0_kwargs, _written_bank
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v6_test import D_KEY, WIDTH

SLOTS = 2
TOP = 4  # tiny patch tokens per camera
STATE_DIM = 2  # the tiny model's action / state width


def _vis_kwargs(**overrides) -> dict:
    values = _v0_kwargs(
        memory_v0920_history_frames=0,
        memory_vis_bank=True,
        memory_vis_slots=SLOTS,
        memory=memory.MemoryConfig(  # the real widths: Pi0Config validates the bank against the PaliGemma width
            d_input=2048, d_key=512, hidden_dims=(), d_value=2048, mlp_l2norm=True, blank_initial_output=True,
            write_rule="delta_output", association_mode="pooled_frame", delta_rate=1.0, alpha_step=0.01,
        ),
    )
    values.update(overrides)
    return values


def test_vis_config_gate_and_defaults():
    on = pi0_config.Pi0Config(**_vis_kwargs())
    assert on.memory_vis_bank and on.memory_vis_slots == SLOTS and on.memory_vis_zero_read is False
    off = pi0_config.Pi0Config(**_v0_kwargs())
    assert off.memory_vis_bank is False and off.memory_vis_slots == 8 and off.memory_vis_input_rms is None
    with pytest.raises(ValueError, match="memory_v0920_input_read"):
        pi0_config.Pi0Config(**_vis_kwargs(memory_v0920_input_read=False))
    with pytest.raises(ValueError, match="memory_v35_enabled"):
        pi0_config.Pi0Config(**_vis_kwargs(memory_v35_enabled=False))
    with pytest.raises(ValueError, match="memory_vis_slots"):
        pi0_config.Pi0Config(**_vis_kwargs(memory_vis_slots=0))
    with pytest.raises(ValueError, match="delta_output"):  # the gradient-rule MLP bank is not the sentence-bank form
        pi0_config.Pi0Config(**_vis_kwargs(memory=memory.MemoryConfig(d_input=2048, d_key=512, hidden_dims=(1024,), d_value=2048)))
    with pytest.raises(ValueError, match="memory_vis_input_rms"):
        pi0_config.Pi0Config(**_vis_kwargs(memory_vis_input_rms=0.0))


class _TinyVis(_TinyV0):
    """The tiny 0920 model (no history frames) with the visual bank: `memory` = a linear delta bank, plus the pooler,
    key/value maps, fixed read queries, gate and slot embeddings, all named memory_vis_*."""

    _vis_input_scale = pi0.Pi0._vis_input_scale
    vis_write_kv = pi0.Pi0.vis_write_kv
    vis_bank_write = pi0.Pi0.vis_bank_write
    vis_read_tokens = pi0.Pi0.vis_read_tokens

    def __init__(self, rngs: nnx.Rngs, *, on: bool = True, image: bool = True, state: bool = False, rule: str = "delta"):
        super().__init__(rngs, history_frames=0)
        self.memory = memory.TitansMemory(
            memory.MemoryConfig(
                d_input=WIDTH, d_key=D_KEY, hidden_dims=(), d_value=WIDTH, mlp_l2norm=True, blank_initial_output=True,
                write_rule="delta_output", association_mode="pooled_frame", delta_rate=1.0, alpha_step=0.01, commit_rule=rule,
            ),
            rngs=rngs,
        )
        self.memory_vis_bank = on
        self.memory_vis_slots = SLOTS
        self.memory_vis_input_rms = None
        self.memory_vis_zero_read = False
        self.memory_vis_image_write = image
        self.memory_vis_state_slot = state
        if image:
            self.memory_vis_pooler = pi0.MemoryQueryCompressor(num_queries=SLOTS, width=WIDTH, num_heads=1, rngs=rngs)
        if state:
            self.memory_vis_state_proj = nnx.Linear(STATE_DIM, WIDTH, rngs=rngs)
            self.memory_vis_state_key = nnx.Param(jax.random.normal(rngs.params(), (D_KEY,), dtype=jnp.float32) / jnp.sqrt(jnp.float32(D_KEY)))
        self.memory_vis_key_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
        self.memory_vis_value_proj = nnx.Linear(WIDTH, WIDTH, use_bias=False, rngs=rngs)
        self.memory_vis_value_proj.kernel.value = jnp.eye(WIDTH, dtype=jnp.float32)
        self.memory_vis_slot_key = nnx.Param(jax.random.normal(rngs.params(), (SLOTS, D_KEY), dtype=jnp.float32) / jnp.sqrt(jnp.float32(D_KEY)))
        self.memory_vis_read_query_bank = nnx.Param(jax.random.normal(rngs.params(), (SLOTS, WIDTH), dtype=jnp.float32) / jnp.sqrt(jnp.float32(WIDTH)))
        self.memory_vis_query_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
        self.memory_vis_inject_w = nnx.Param(jnp.full((WIDTH,), jnp.arctanh(jnp.float32(0.5)), dtype=jnp.float32))
        self.memory_vis_slot_embedding = nnx.Param(jnp.zeros((SLOTS, WIDTH), dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny_vis():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyVis(nnx.Rngs(9))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _tiny(**kwargs):
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        return _TinyVis(nnx.Rngs(9), **kwargs)
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


@pytest.fixture(scope="module")
def tiny_vis_state():
    return _tiny(image=True, state=True)


@pytest.fixture(scope="module")
def tiny_state_only_add():
    return _tiny(image=False, state=True, rule="additive")


@pytest.fixture(scope="module")
def tiny_snap():
    """The original tiny 0920 class with the same seed and no history frames: what the flag-off model must reproduce."""
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV0(nnx.Rngs(9), history_frames=0)
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _front(model, observation):
    prefix, mask, ar = model.embed_prefix(observation)
    return prefix, mask, ar, prefix[:, :TOP]


def _written_visual(model, front, steps: int = 1):
    state = model.memory.init_state(front.shape[0])
    keys, values, _ = model.vis_write_kv(front)
    for _ in range(steps):
        state, _ = model.vis_bank_write(state, keys, values, jnp.ones((front.shape[0],), dtype=bool))
    return state, keys, values


def _main_terms(losses):
    return {k: np.asarray(losses[k]) for k in ("ce", "flow", "v4_decision_ce_steps", "v4_sem_commit_count", "v4_use_flow_sum")}


# --------------------------------------------------------------------------- (b) flag off == snap


def test_flag_off_is_snap_same_tokens_and_losses(tiny_vis, tiny_snap):
    model = tiny_vis
    step0 = _single_step_observation()
    seq = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    model.memory_vis_bank = False
    try:
        assert model._memory_token_total == tiny_snap._memory_token_total == 3
        prefix, mask, ar, _ = _front(model, step0)
        blank, written = _written_bank(model)
        off = model._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=TOP, visual_state=model.memory.init_state(1))
        ref = tiny_snap._v0920_prepare_prefix(*_front(tiny_snap, step0)[:3], written, top_token_count=TOP)
        assert off["memory_tokens"].shape == ref["memory_tokens"].shape == (1, 3, WIDTH)
        np.testing.assert_array_equal(np.asarray(off["final_prefix"]), np.asarray(ref["final_prefix"]))
        assert "vis_keys" not in off
        losses_off = model._compute_sequence_loss_v32(jax.random.key(922), seq, actions, train=False)
        losses_ref = tiny_snap._compute_sequence_loss_v32(jax.random.key(922), seq, actions, train=False)
        for key, value in _main_terms(losses_off).items():
            np.testing.assert_array_equal(value, _main_terms(losses_ref)[key], err_msg=key)
        for key in ("vis_commit_count", "vis_raw_read_rms_sum", "vis_injected_pre_cast_rms_sum", "vis_bank_norm_sum"):
            assert float(losses_off[key]) == 0.0 and float(losses_ref[key]) == 0.0
    finally:
        model.memory_vis_bank = True


# --------------------------------------------------------------------------- (c) the read


def test_empty_visual_bank_reads_zero_and_a_written_bank_reads_at_the_image_scale(tiny_vis):
    model = tiny_vis
    _, _, _, front = _front(model, _single_step_observation())
    blank = model.memory.init_state(1)
    tokens, valid, retrieved, queries, pre, post = model.vis_read_tokens(blank, front, jnp.float32)
    assert tokens.shape == (1, SLOTS, WIDTH) and queries.shape == (1, SLOTS, D_KEY)
    np.testing.assert_array_equal(np.asarray(tokens), 0.0)
    assert not bool(np.any(np.asarray(valid))) and float(pre[0]) == 0.0
    np.testing.assert_allclose(np.linalg.norm(np.asarray(queries), axis=-1), 1.0, atol=1e-5)

    written, keys, values = _written_visual(model, front)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(keys), axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(values), axis=-1), 1.0, atol=1e-5)
    tokens_w, valid_w, retrieved_w, queries_w, pre_w, post_w = model.vis_read_tokens(written, front, jnp.float32)
    assert bool(np.all(np.asarray(valid_w))) and np.all(np.isfinite(np.asarray(tokens_w)))
    np.testing.assert_array_equal(np.asarray(queries_w), np.asarray(queries))  # fixed queries: the same on any bank
    # every token sits at gate * target: tanh(w) = 0.5 at init, target = the RMS of this sample's front image tokens
    target = float(np.sqrt(np.mean(np.square(np.asarray(front, dtype=np.float32)))))
    per_token = np.sqrt(np.mean(np.square(np.asarray(tokens_w)), axis=-1))
    np.testing.assert_allclose(per_token, 0.5 * target, rtol=1e-4)
    assert float(pre_w[0]) > 0.0 and float(post_w[0]) > 0.0
    # a fixed target overrides the image scale; the zero-read switch and zero_read both silence the read
    model.memory_vis_input_rms = 2.0
    try:
        tokens_f = model.vis_read_tokens(written, front, jnp.float32)[0]
        np.testing.assert_allclose(np.sqrt(np.mean(np.square(np.asarray(tokens_f)), axis=-1)), 1.0, rtol=1e-4)
    finally:
        model.memory_vis_input_rms = None
    np.testing.assert_array_equal(np.asarray(model.vis_read_tokens(written, front, jnp.float32, zero_read=True)[0]), 0.0)
    model.memory_vis_zero_read = True
    try:
        tokens_z, valid_z = model.vis_read_tokens(written, front, jnp.float32)[:2]
        np.testing.assert_array_equal(np.asarray(tokens_z), 0.0)
        assert not bool(np.any(np.asarray(valid_z)))
    finally:
        model.memory_vis_zero_read = False


def test_prefix_pass_appends_sentence_then_visual_tokens_and_tick_zero_equals_snap(tiny_vis):
    model = tiny_vis
    step0 = _single_step_observation()
    prefix, mask, ar, front = _front(model, step0)
    blank_sem, written_sem = _written_bank(model)
    blank_vis = model.memory.init_state(1)
    written_vis, _, _ = _written_visual(model, front)
    assert model._memory_token_total == 3 + SLOTS
    with pytest.raises(ValueError, match="visual_state"):
        model._v0920_prepare_prefix(prefix, mask, ar, written_sem, top_token_count=TOP)
    p = model._v0920_prepare_prefix(prefix, mask, ar, written_sem, top_token_count=TOP, visual_state=written_vis)
    assert p["memory_tokens"].shape == (1, 3 + SLOTS, WIDTH) and p["memory_valid"].shape == (1, 3 + SLOTS)
    assert p["final_prefix"].shape == (1, mask.shape[1] + 3 + SLOTS, WIDTH)
    assert p["cache"][0].shape[2] == mask.shape[1] + 3 + SLOTS + model.causal_token_len
    assert p["vis_keys"].shape == (1, SLOTS, D_KEY) and p["vis_values"].shape == (1, SLOTS, WIDTH)
    # the v3.5 machinery still sees snap's zero write tokens / zero retrieval
    np.testing.assert_array_equal(np.asarray(p["write_tokens"]), 0.0)
    np.testing.assert_array_equal(np.asarray(p["retrieved"]), 0.0)
    assert bool(np.all(np.asarray(p["memory_valid"])))
    # an EMPTY visual bank: its tokens are masked and the prefix rows + sentence tokens equal snap's exactly
    p0 = model._v0920_prepare_prefix(prefix, mask, ar, written_sem, top_token_count=TOP, visual_state=blank_vis)
    assert not bool(np.any(np.asarray(p0["memory_valid"][:, 3:])))
    model.memory_vis_bank = False
    try:
        snap = model._v0920_prepare_prefix(prefix, mask, ar, written_sem, top_token_count=TOP)
    finally:
        model.memory_vis_bank = True
    n = mask.shape[1] + 3
    np.testing.assert_allclose(np.asarray(p0["final_prefix"][:, :n]), np.asarray(snap["final_prefix"][:, :n]), atol=1e-6)
    # a WRITTEN visual bank changes the prefix rows (the visual columns are keys from block 0)
    assert not np.allclose(np.asarray(p["final_prefix"][:, : mask.shape[1]]), np.asarray(snap["final_prefix"][:, : mask.shape[1]]), atol=1e-6)


# --------------------------------------------------------------------------- (d) memory-blind, stop-gradient write


def test_write_content_is_stop_gradient_and_only_trains_the_visual_leaves(tiny_vis):
    model = tiny_vis
    step0 = _single_step_observation()

    def write_loss(m):
        prefix, _, _ = m.embed_prefix(step0)
        keys, values, pooled = m.vis_write_kv(prefix[:, :TOP])
        return jnp.sum(keys) + jnp.sum(values) + jnp.sum(pooled)

    grads = nnx.grad(write_loss)(model)
    flat = jax.tree_util.tree_leaves_with_path(grads)
    touched = sorted({"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path).split("/")[0] for path, leaf in flat if float(jnp.max(jnp.abs(leaf))) > 0.0})
    assert touched, "the write must train its own parameters"
    assert all(name.startswith("memory_vis_") for name in touched), touched
    assert "memory_vis_pooler" in touched and "memory_vis_key_proj" in touched and "memory_vis_slot_key" in touched


# --------------------------------------------------------------------------- (e) the transition


def test_transition_commits_decays_and_keeps_associations(tiny_vis):
    model = tiny_vis
    _, _, _, front = _front(model, _single_step_observation())
    blank = model.memory.init_state(1)
    keys, values, _ = model.vis_write_kv(front)
    # commit False == exactly one analytic decay step (on a written state too)
    written, _, _ = _written_visual(model, front)
    decayed, aux_no = model.vis_bank_write(written, keys, values, jnp.zeros((1,), dtype=bool))
    ref, _ = model.memory.analytic_decay(written, 1)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), decayed, ref)
    assert not bool(np.any(np.asarray(aux_no["commit_applied"])))
    # a blank bank with commit False stays exactly blank
    still, _ = model.vis_bank_write(blank, keys, values, jnp.zeros((1,), dtype=bool))
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), still, blank)
    # a commit stores every slot: read back with the write keys ~ the values
    w1, aux = model.vis_bank_write(blank, keys, values, jnp.ones((1,), dtype=bool))
    assert bool(np.all(np.asarray(aux["commit_applied"])))
    back = np.asarray(model.memory.read_key(w1, keys))[0]
    cos = np.sum(back * np.asarray(values)[0], axis=-1) / np.maximum(np.linalg.norm(back, axis=-1), 1e-6)
    assert np.all(cos > 0.9), cos
    # a second, different association (another frame) does not destroy the first
    keys2, values2, _ = model.vis_write_kv(front * 0.3 + 0.7 * jnp.flip(front, axis=1))
    w2, _ = model.vis_bank_write(w1, keys2, values2, jnp.ones((1,), dtype=bool))
    back1 = np.asarray(model.memory.read_key(w2, keys))[0]
    cos1 = np.sum(back1 * np.asarray(values)[0], axis=-1) / np.maximum(np.linalg.norm(back1, axis=-1), 1e-6)
    assert np.all(cos1 > 0.5), cos1
    # re-committing the same content is nearly a no-op (delta rule): the state moves far less than the first commit
    w1b, _ = model.vis_bank_write(w1, keys, values, jnp.ones((1,), dtype=bool))
    name = model.memory._output_weight_name
    first = float(jnp.linalg.norm(w1.fast_weights[name]))
    again = float(jnp.linalg.norm(w1b.fast_weights[name] - w1.fast_weights[name]))
    assert again < 0.05 * first, (again, first)


# --------------------------------------------------------------------------- (f) the sequence loss


def test_sequence_loss_commits_every_valid_tick_trains_the_visual_leaves_and_is_finite(tiny_vis):
    model = tiny_vis
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    np.testing.assert_array_equal(losses["vis_commit_count"], 3.0)  # every valid tick
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)  # the sentence path is untouched
    assert float(losses["vis_bank_norm_sum"]) > 0.0
    assert float(losses["vis_raw_read_rms_sum"]) > 0.0  # ticks 1 and 2 read a written bank
    assert float(losses["vis_injected_pre_cast_rms_sum"]) > 0.0
    # an invalid tick neither commits nor counts
    partial = observation.replace(seq_step_mask=jnp.asarray([[True, False, True]]))
    losses_p = model._compute_sequence_loss_v32(jax.random.key(922), partial, actions, train=False)
    np.testing.assert_array_equal(losses_p["vis_commit_count"], 2.0)

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["ce"]) + jnp.sum(out["flow"])

    grads = nnx.grad(total_loss)(model)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads) if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    for name in ("memory_vis_read_query_bank", "memory_vis_query_proj", "memory_vis_inject_w", "memory_vis_slot_embedding",
                 "memory_vis_key_proj", "memory_vis_value_proj", "memory_vis_slot_key", "memory_vis_pooler"):
        leaves = jax.tree_util.tree_leaves(grads[name])
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves) > 0.0, name
    # the sentence read is still trained too
    assert float(jnp.max(jnp.abs(grads["memory_sem_read_query_bank"].value))) > 0.0


def test_zeroed_visual_read_cannot_leak_the_write_and_stays_next_to_snap(tiny_vis):
    """With the visual read silenced the bank still writes every tick, but nothing of it may reach the loss: the trained
    terms must not move when the WRITE content is changed (perturbed pooler queries -> different keys / values / bank), and
    they stay within a fraction of a percent of the flag-off (snap) terms -- the only remaining difference is that the
    sentence / action tokens sit `SLOTS` RoPE positions later, as they do for any appended token."""
    model = tiny_vis
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    model.memory_vis_zero_read = True
    try:
        zeroed = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
        saved = model.memory_vis_pooler.query_bank.value
        model.memory_vis_pooler.query_bank.value = saved * -3.0 + 0.5
        try:
            perturbed = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
        finally:
            model.memory_vis_pooler.query_bank.value = saved
    finally:
        model.memory_vis_zero_read = False
    model.memory_vis_bank = False
    try:
        snap = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    finally:
        model.memory_vis_bank = True
    np.testing.assert_array_equal(zeroed["vis_commit_count"], 3.0)  # the bank kept writing
    assert float(zeroed["vis_bank_norm_sum"]) != float(perturbed["vis_bank_norm_sum"])  # ... different content
    for key, value in _main_terms(zeroed).items():
        np.testing.assert_array_equal(value, _main_terms(perturbed)[key], err_msg=key)  # ... nothing of it in the loss
    for key, value in _main_terms(zeroed).items():
        np.testing.assert_allclose(value, _main_terms(snap)[key], rtol=1e-2, atol=1e-6, err_msg=key)


# --------------------------------------------------------------------------- (g) the sampler


def test_sampler_advances_the_visual_bank_per_tick_in_normal_mode_only(tiny_vis):
    from openpi.models.pi0_v35_test import _single_observation

    model = tiny_vis
    observation = _single_observation()
    blank_sem, written_sem = _written_bank(model)
    visual = model.memory.init_state(1)
    name = model.memory._output_weight_name
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32)}
    # frozen (what the server passes for snap): the visual bank does not move
    _, frozen, _ = model.sample_with_memory(jax.random.key(1), observation, visual, semantic_state=written_sem, write_mode="frozen", **kwargs)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), frozen, visual)
    # normal + the v3.5 masks (what the server passes for a visual-bank model): one commit per served tick
    _, s1, aux1 = model.sample_with_memory(
        jax.random.key(1), observation, visual, semantic_state=written_sem, write_mode="normal",
        v35_transition_valid=True, v35_write_mask=True, **kwargs,
    )
    assert float(jnp.linalg.norm(s1.fast_weights[name])) > 0.0
    _, s2, aux2 = model.sample_with_memory(
        jax.random.key(1), observation, s1, semantic_state=written_sem, write_mode="normal",
        v35_transition_valid=True, v35_write_mask=True, **kwargs,
    )
    # the second tick READ a written bank: its outputs differ from the first tick's (same rng, same observation)
    assert not (
        np.array_equal(np.asarray(aux1["tokens"]), np.asarray(aux2["tokens"]))
        and np.allclose(np.asarray(aux1["token_prob"]), np.asarray(aux2["token_prob"]))
    )
    # normal WITHOUT the masks is a no-op (the v3.5 fail-closed contract), dynamics_only is exactly one decay
    _, noop, _ = model.sample_with_memory(jax.random.key(1), observation, s1, semantic_state=written_sem, write_mode="normal", **kwargs)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), noop, s1)
    _, dyn, _ = model.sample_with_memory(
        jax.random.key(1), observation, s1, semantic_state=written_sem, write_mode="dynamics_only",
        v35_transition_valid=True, v35_write_mask=True, **kwargs,
    )
    ref, _ = model.memory.analytic_decay(s1, 1)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6), dyn, ref)


# --------------------------------------------------------------------------- (h) the additive rule, the state slot


def _bank(rule: str):
    return memory.TitansMemory(
        memory.MemoryConfig(
            d_input=WIDTH, d_key=D_KEY, hidden_dims=(), d_value=WIDTH, mlp_l2norm=True, blank_initial_output=True,
            write_rule="delta_output", association_mode="pooled_frame", delta_rate=1.0, alpha_step=0.01, commit_rule=rule,
        ),
        rngs=nnx.Rngs(3),
    )


def test_additive_rule_accumulates_repeats_and_the_delta_rule_does_not():
    with pytest.raises(ValueError, match="commit_rule"):
        memory.MemoryConfig(commit_rule="hebb")
    with pytest.raises(ValueError, match="additive"):
        memory.MemoryConfig(commit_rule="additive")  # needs the delta_output form
    k = memory.l2_normalize(jax.random.normal(jax.random.key(1), (1, 1, D_KEY)))
    v = memory.l2_normalize(jax.random.normal(jax.random.key(2), (1, 1, WIDTH)))
    ones = jnp.ones((1, 1), dtype=bool)
    reads = {}
    for rule in ("delta", "additive"):
        bank = _bank(rule)
        state = bank.init_state(1)
        for _ in range(3):
            state, aux = bank.delta_write_kv_multi(state, k, v, ones)
            assert bool(aux["commit_applied"][0, 0])
        reads[rule] = np.asarray(bank.read_key(state, k))[0, 0]
    unit_v = np.asarray(v)[0, 0]
    # delta: the bank returns v (refreshed against the decay), additive: v + 0.99 v + 0.99^2 v
    np.testing.assert_allclose(reads["delta"], unit_v, atol=1e-4)
    np.testing.assert_allclose(reads["additive"], (1 + 0.99 + 0.99**2) * unit_v, atol=1e-4)
    # the default is delta, bit-for-bit
    assert memory.MemoryConfig(write_rule="delta_output", association_mode="pooled_frame").commit_rule == "delta"


def test_state_slot_is_one_more_association_from_the_state_only(tiny_vis_state):
    model = tiny_vis_state
    step0 = _single_step_observation()
    prefix, _, _, front = _front(model, step0)
    state = step0.state
    keys, values, pooled = model.vis_write_kv(front, state=state)
    assert keys.shape == (1, SLOTS + 1, D_KEY) and values.shape == (1, SLOTS + 1, WIDTH) and pooled.shape == (1, SLOTS + 1, WIDTH)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(keys), axis=-1), 1.0, atol=1e-5)
    with pytest.raises(ValueError, match="state"):
        model.vis_write_kv(front)
    # a different state changes the state slot only; a different image changes the image slots only
    keys_s, values_s, _ = model.vis_write_kv(front, state=state + 1.0)
    np.testing.assert_array_equal(np.asarray(keys_s[:, :SLOTS]), np.asarray(keys[:, :SLOTS]))
    assert not np.allclose(np.asarray(keys_s[:, SLOTS]), np.asarray(keys[:, SLOTS]), atol=1e-4)
    keys_i, _, _ = model.vis_write_kv(front * 0.5, state=state)
    np.testing.assert_array_equal(np.asarray(keys_i[:, SLOTS]), np.asarray(keys[:, SLOTS]))
    assert not np.allclose(np.asarray(keys_i[:, :SLOTS]), np.asarray(keys[:, :SLOTS]), atol=1e-4)

    def write_loss(m):
        p, _, _ = m.embed_prefix(step0)
        k, v, _ = m.vis_write_kv(p[:, :TOP], state=step0.state)
        return jnp.sum(k) + jnp.sum(v)

    grads = nnx.grad(write_loss)(model)
    for name in ("memory_vis_state_proj", "memory_vis_state_key", "memory_vis_pooler"):
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grads[name])) > 0.0, name
    # the prefix builder passes the state through; without it the state slot fails loudly
    blank_sem, written_sem = _written_bank(model)
    with pytest.raises(ValueError, match="state"):
        model._v0920_prepare_prefix(prefix, *_front(model, step0)[1:3], written_sem, top_token_count=TOP, visual_state=model.memory.init_state(1))
    p = model._v0920_prepare_prefix(prefix, *_front(model, step0)[1:3], written_sem, top_token_count=TOP, visual_state=model.memory.init_state(1), state=state)
    assert p["vis_keys"].shape == (1, SLOTS + 1, D_KEY) and p["memory_tokens"].shape == (1, 3 + SLOTS, WIDTH)  # read tokens unchanged


def test_state_only_bank_ignores_the_images_and_keeps_the_read(tiny_state_only_add):
    model = tiny_state_only_add
    step0 = _single_step_observation()
    _, _, _, front = _front(model, step0)
    assert not hasattr(model, "memory_vis_pooler") and model.memory.config.commit_rule == "additive"
    keys, values, _ = model.vis_write_kv(front, state=step0.state)
    assert keys.shape == (1, 1, D_KEY)
    keys_i, _, _ = model.vis_write_kv(front * 3.0, state=step0.state)
    np.testing.assert_array_equal(np.asarray(keys_i), np.asarray(keys))
    assert model._memory_token_total == 3 + SLOTS  # still 8-style read tokens
    # repeats accumulate: three identical ticks read back ~ (1 + 0.99 + 0.99^2) x the value
    state = model.memory.init_state(1)
    for _ in range(3):
        state, _ = model.vis_bank_write(state, keys, values, jnp.ones((1,), dtype=bool))
    back = np.asarray(model.memory.read_key(state, keys))[0, 0]
    np.testing.assert_allclose(back, (1 + 0.99 + 0.99**2) * np.asarray(values)[0, 0], atol=1e-4)
    # ... and the read token is still RMS-matched (the tally shows up as direction, not scale)
    tokens, valid = model.vis_read_tokens(state, front, jnp.float32)[:2]
    target = float(np.sqrt(np.mean(np.square(np.asarray(front, dtype=np.float32)))))
    np.testing.assert_allclose(np.sqrt(np.mean(np.square(np.asarray(tokens)), axis=-1)), 0.5 * target, rtol=1e-4)
    assert bool(np.all(np.asarray(valid)))


@pytest.mark.parametrize("fixture", ["tiny_vis_state", "tiny_state_only_add"])
def test_state_rows_train_and_serve(fixture, request):
    from openpi.models.pi0_v35_test import _single_observation

    model = request.getfixturevalue(fixture)
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    np.testing.assert_array_equal(losses["vis_commit_count"], 3.0)
    assert float(losses["vis_bank_norm_sum"]) > 0.0

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["ce"]) + jnp.sum(out["flow"])

    grads = nnx.grad(total_loss)(model)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads) if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    for name in ("memory_vis_state_proj", "memory_vis_state_key", "memory_vis_read_query_bank"):
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grads[name])) > 0.0, name
    # serving: the bank advances per tick (state slot included) in normal mode
    single = _single_observation()
    blank_sem, written_sem = _written_bank(model)
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32)}
    _, s1, _ = model.sample_with_memory(jax.random.key(1), single, model.memory.init_state(1), semantic_state=written_sem,
                                        write_mode="normal", v35_transition_valid=True, v35_write_mask=True, **kwargs)
    assert float(jnp.linalg.norm(s1.fast_weights[model.memory._output_weight_name])) > 0.0


# --------------------------------------------------------------------------- (i) the v3 snap (beans0922, 09-22 15:30)
# snap moved to v3 = change-only confident own writes (memory_v7_write_every_step False, memory_v5_write_conf 0.9), the
# question context (memory_v0920_query_context) and the error-driven token weight. The sensory bank must be unaffected:
# it still writes every valid tick, and with the flag off the model is the v3 snap.


class _TinyVisV3(_TinyVis):
    """_TinyVis under the v3 flags, with the question-context modules (zero-initialised maps, as in the real model)."""

    def __init__(self, rngs: nnx.Rngs, *, on: bool = True, image: bool = True, state: bool = False, rule: str = "delta"):
        super().__init__(rngs, on=on, image=image, state=state, rule=rule)
        _apply_v3_flags(self)
        r = self.memory_sem_read_query_bank.value.shape[0]
        self.memory_v0920_query_context = True
        self.memory_sem_query_context_pooler = pi0.MemoryQueryCompressor(num_queries=r, width=WIDTH, num_heads=2, rngs=rngs)
        self.memory_sem_query_context_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.memory_sem_query_prev_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)


def _apply_v3_flags(model) -> None:
    model.memory_v7_write_every_step = False
    model.memory_v5_write_conf = 0.9
    model.memory_v7_hard_token_ce_weight = 5.0


def _v3_note(model):
    """A committed two-token note as prev_tokens / prev_mask [1, sentence_len] (the question context needs one)."""
    length = int(model.memory_v5_sentence_len) if hasattr(model, "memory_v5_sentence_len") else 4
    tokens = jnp.zeros((1, length), dtype=jnp.int32).at[0, 0].set(5).at[0, 1].set(6)
    mask = jnp.zeros((1, length), dtype=bool).at[0, :2].set(True)
    return tokens, mask


@pytest.fixture(scope="module")
def tiny_vis_v3():
    return _TinyVisV3(nnx.Rngs(922))


def test_v3_flag_off_is_the_v3_snap(tiny_vis_v3):
    from openpi.models.pi0_v0920_query_context_test import _TinyV0Ctx

    model = tiny_vis_v3
    ref = _TinyV0Ctx(nnx.Rngs(922))  # the same shared leaves (same rng stream first), context maps zero -> no shift
    _apply_v3_flags(ref)
    step0 = _single_step_observation()
    seq = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    prev_tokens, prev_mask = _v3_note(model)
    model.memory_vis_bank = False
    try:
        prefix, mask, ar, _ = _front(model, step0)
        _, written = _written_bank(model)
        off = model._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=TOP, visual_state=model.memory.init_state(1),
                                          prev_tokens=prev_tokens, prev_mask=prev_mask)
        on = ref._v0920_prepare_prefix(*_front(ref, step0)[:3], written, top_token_count=TOP, prev_tokens=prev_tokens, prev_mask=prev_mask)
        np.testing.assert_array_equal(np.asarray(off["final_prefix"]), np.asarray(on["final_prefix"]))
        losses_off = model._compute_sequence_loss_v32(jax.random.key(922), seq, actions, train=False)
        losses_ref = ref._compute_sequence_loss_v32(jax.random.key(922), seq, actions, train=False)
        for key, value in _main_terms(losses_off).items():
            np.testing.assert_array_equal(value, _main_terms(losses_ref)[key], err_msg=key)
        np.testing.assert_array_equal(losses_off["v4_sem_commit_count"], losses_ref["v4_sem_commit_count"])
    finally:
        model.memory_vis_bank = True


def test_v3_sensory_bank_still_writes_every_valid_tick_while_the_sentence_writes_are_gated(tiny_vis_v3):
    model = tiny_vis_v3
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    np.testing.assert_array_equal(losses["vis_commit_count"], losses["vis_valid_count"])  # gated on tick validity only
    np.testing.assert_array_equal(losses["vis_commit_count"], 3.0)
    assert 0.0 <= float(losses["v4_sem_commit_count"]) <= 3.0  # the sentence bank: changed AND confident notes only
    assert float(losses["vis_bank_norm_sum"]) > 0.0
    partial = observation.replace(seq_step_mask=jnp.asarray([[True, False, True]]))
    np.testing.assert_array_equal(model._compute_sequence_loss_v32(jax.random.key(922), partial, actions, train=False)["vis_commit_count"], 2.0)
    # gradients: with the ramp's label writes (always confident, as in the first 500 updates) the sentence bank has content,
    # so the sentence read, the two v3 question maps AND every sensory leaf train together
    labelled = observation.replace(seq_label_write_prob=jnp.ones((1,), dtype=jnp.float32))
    np.testing.assert_array_equal(model._compute_sequence_loss_v32(jax.random.key(922), labelled, actions, train=False)["v4_sem_commit_count"], 3.0)

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(922), labelled, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["ce"]) + jnp.sum(out["flow"])

    grads = nnx.grad(total_loss)(model)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads) if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    for name in ("memory_vis_read_query_bank", "memory_vis_query_proj", "memory_vis_key_proj", "memory_vis_value_proj",
                 "memory_vis_slot_key", "memory_vis_pooler", "memory_sem_read_query_bank", "memory_sem_query_context_proj",
                 "memory_sem_query_prev_proj"):
        leaves = jax.tree_util.tree_leaves(grads[name])
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves) > 0.0, name
    # own writes only (after the ramp): the sensory leaves still train even when no note passes the 0.9 gate
    def own_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["ce"]) + jnp.sum(out["flow"])

    own = nnx.grad(own_loss)(model)
    assert max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(own["memory_vis_pooler"])) > 0.0
    assert float(jnp.max(jnp.abs(own["memory_vis_read_query_bank"].value))) > 0.0


def test_v3_sampler_advances_the_sensory_bank_with_the_note_context(tiny_vis_v3):
    from openpi.models.pi0_v35_test import _single_observation

    model = tiny_vis_v3
    observation = _single_observation()
    _, written_sem = _written_bank(model)
    prev_tokens, prev_mask = _v3_note(model)
    visual = model.memory.init_state(1)
    name = model.memory._output_weight_name
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32),
              "v5_prev_tokens": prev_tokens, "v5_prev_mask": prev_mask}
    _, s1, _ = model.sample_with_memory(
        jax.random.key(1), observation, visual, semantic_state=written_sem, write_mode="normal",
        v35_transition_valid=True, v35_write_mask=True, **kwargs,
    )
    assert float(jnp.linalg.norm(s1.fast_weights[name])) > 0.0  # the served tick wrote the sensory bank
    _, frozen, _ = model.sample_with_memory(jax.random.key(1), observation, visual, semantic_state=written_sem, write_mode="frozen", **kwargs)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), frozen, visual)
