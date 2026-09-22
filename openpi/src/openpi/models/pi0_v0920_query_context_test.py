"""0920_v2 "look before you ask" (Pi0Config.memory_v0920_query_context) on the tiny 0920 stand-in: (a) the config gate;
(b) at init the questions, the read tokens and the whole prefix pass are BIT-IDENTICAL to the fixed-question model (the two
context maps start at zero); (c) once the maps are non-zero the questions depend on the frame and on the last note, and an
empty note shifts nothing; (d) the missing-argument guard; (e) the sampler path runs with the previous-note arguments; (f) the
context maps receive gradient through the read."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v0920_test import _TinyV0, _single_step_observation, _v0_kwargs, _written_bank
from openpi.models.pi0_v6_test import D_KEY, WIDTH


class _TinyV0Ctx(_TinyV0):
    """The tiny 0920 model without history frames, with the question-context modules when `context` is set."""

    def __init__(self, rngs: nnx.Rngs, *, context: bool = True):
        super().__init__(rngs, history_frames=0)
        self.memory_v0920_query_context = context
        if context:
            r = self.memory_sem_read_query_bank.value.shape[0]
            self.memory_sem_query_context_pooler = pi0.MemoryQueryCompressor(num_queries=r, width=WIDTH, num_heads=2, rngs=rngs)
            self.memory_sem_query_context_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)
            self.memory_sem_query_prev_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)


@pytest.fixture(scope="module")
def pair():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        # the same seed: every shared leaf is identical, the context modules are drawn afterwards
        yield _TinyV0Ctx(nnx.Rngs(9), context=False), _TinyV0Ctx(nnx.Rngs(9), context=True)
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _note(model, tokens=(5, 6)):
    length = model.memory_v5_sentence_len
    row = np.zeros((1, length), dtype=np.int32)
    row[0, : len(tokens)] = tokens
    return jnp.asarray(row), jnp.asarray(np.arange(length)[None] < len(tokens))


def test_query_context_config_gate():
    on = pi0_config.Pi0Config(**_v0_kwargs(memory_v0920_history_frames=0, memory_v0920_query_context=True))
    assert on.memory_v0920_query_context
    assert not pi0_config.Pi0Config(**_v0_kwargs()).memory_v0920_query_context
    with pytest.raises(ValueError, match="memory_v0920_query_context"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v0920_input_read=False, memory_v7_no_visual_block=False, memory_v0920_query_context=True))


def test_zero_init_context_is_bit_identical_to_the_fixed_questions(pair):
    fixed, ctx = pair
    step0 = _single_step_observation()
    prefix, mask, ar = fixed.embed_prefix(step0)
    _, written = _written_bank(fixed)
    prev_tokens, prev_mask = _note(ctx)
    a = fixed._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4)
    b = ctx._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4, prev_tokens=prev_tokens, prev_mask=prev_mask)
    np.testing.assert_array_equal(np.asarray(a["sem_queries"]), np.asarray(b["sem_queries"]))
    np.testing.assert_array_equal(np.asarray(a["memory_tokens"]), np.asarray(b["memory_tokens"]))
    np.testing.assert_array_equal(np.asarray(a["final_prefix"]), np.asarray(b["final_prefix"]))


def test_questions_follow_the_frame_and_the_note_once_the_maps_open(pair):
    _, ctx = pair
    step0 = _single_step_observation()
    prefix, mask, ar = ctx.embed_prefix(step0)
    _, written = _written_bank(ctx)
    prev_tokens, prev_mask = _note(ctx)
    ctx_open = nnx.clone(ctx)
    ctx_open.memory_sem_query_context_proj.kernel.value = 0.05 * jnp.ones((WIDTH, D_KEY), dtype=jnp.float32)
    ctx_open.memory_sem_query_prev_proj.kernel.value = 0.05 * jnp.ones((WIDTH, D_KEY), dtype=jnp.float32)

    def queries(m, p, pt, pm):
        return np.asarray(m.v0920_read_tokens(written, 1, p.dtype, context=p, context_valid=mask, prev_tokens=pt, prev_mask=pm)[3])

    q0 = queries(ctx, prefix, prev_tokens, prev_mask)
    q_open = queries(ctx_open, prefix, prev_tokens, prev_mask)
    assert not np.allclose(q0, q_open, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(q_open, axis=-1), 1.0, atol=1e-5)  # still unit questions
    # another frame -> other questions
    q_other = queries(ctx_open, prefix * 0.5 + 0.1, prev_tokens, prev_mask)
    assert not np.allclose(q_open, q_other, atol=1e-6)
    # another note -> other questions; an EMPTY note contributes nothing (the frame term alone remains)
    q_note2 = queries(ctx_open, prefix, *_note(ctx, (7, 8)))
    assert not np.allclose(q_open, q_note2, atol=1e-6)
    empty_tokens, empty_mask = _note(ctx, ())
    q_empty = queries(ctx_open, prefix, empty_tokens, empty_mask)
    ctx_frame_only = nnx.clone(ctx_open)
    ctx_frame_only.memory_sem_query_prev_proj.kernel.value = jnp.zeros((WIDTH, D_KEY), dtype=jnp.float32)
    np.testing.assert_allclose(q_empty, queries(ctx_frame_only, prefix, prev_tokens, prev_mask), atol=1e-6)


def test_missing_note_arguments_are_refused(pair):
    _, ctx = pair
    step0 = _single_step_observation()
    prefix, mask, ar = ctx.embed_prefix(step0)
    _, written = _written_bank(ctx)
    with pytest.raises(ValueError, match="memory_v0920_query_context"):
        ctx._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4)


def test_sampler_runs_with_the_previous_note(pair):
    _, ctx = pair
    observation = _single_step_observation()
    visual = ctx.memory.init_state(1)
    _, written = _written_bank(ctx)
    prev_tokens, prev_mask = _note(ctx)
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32), "write_mode": "frozen"}
    _, _, aux = ctx.sample_with_memory(jax.random.key(923), observation, visual, semantic_state=written,
                                       v5_prev_tokens=prev_tokens, v5_prev_mask=prev_mask, **kwargs)
    assert aux["sem_queries"].shape == (1, 3, D_KEY)
    assert np.all(np.isfinite(np.asarray(aux["token_prob"])))


def test_context_maps_receive_gradient_through_the_read(pair):
    _, ctx = pair
    step0 = _single_step_observation()
    _, written = _written_bank(ctx)
    prev_tokens, prev_mask = _note(ctx)

    def read_loss(m):
        p, mk, a = m.embed_prefix(step0)
        out = m._v0920_prepare_prefix(p, mk, a, written, top_token_count=4, prev_tokens=prev_tokens, prev_mask=prev_mask)
        return jnp.sum(out["final_prefix"])

    grads = nnx.grad(read_loss)(ctx)
    assert float(jnp.max(jnp.abs(grads["memory_sem_query_context_proj"]["kernel"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_query_prev_proj"]["kernel"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_read_query_bank"].value))) > 0.0


def test_sequence_loss_runs_through_the_training_scan_with_the_note_context(pair):
    from openpi.models.pi0_v4_test import _v4_sequence_observation

    _, ctx = pair
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = ctx._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(ctx)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads) if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    # the scan passes the last committed note and the prefix into the questions: both context maps get gradient
    assert float(jnp.max(jnp.abs(grads["memory_sem_query_context_proj"]["kernel"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_read_query_bank"].value))) > 0.0
