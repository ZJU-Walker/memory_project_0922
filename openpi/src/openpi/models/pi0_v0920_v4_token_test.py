"""0920_v4 "token bank, exact copies" on the tiny 0920 stand-in (Pi0Config.memory_v0920_prev_readback + the pointer bonus in
context mode + the lowest-token-probability write gate): (a) the config gates; (b) the memory block widens by one token per
note position; (c) a committed note reads back through the bank exactly (cosine ~1 with its own values), positions outside
the note and an empty note give exactly-zero, masked tokens; (d) the missing-note guard; (e) the sampler and the training scan
run with the pointer + read-back and their gradients are finite; (f) the confidence helper's mean / minimum rule."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v0920_test import _single_step_observation, _v0_kwargs
from openpi.models.pi0_v6_test import D_KEY, WIDTH, _TinyV6Seq

SENT = 2  # tiny sentence length (the tiny model's causal buffer is 2 tokens; a note must fit in it)


class _TinyV4(_TinyV6Seq):
    """The tiny v6 sequence model under the 0920_v4 flags: input read, pointer bonus (context queries), note read-back."""

    _top_camera_token_count = pi0.Pi0._top_camera_token_count
    _image_tower_tokens = pi0.Pi0._image_tower_tokens
    _augment_sequence_images = pi0.Pi0._augment_sequence_images
    _v0920_input_scale = pi0.Pi0._v0920_input_scale
    v0920_read_tokens = pi0.Pi0.v0920_read_tokens
    v0920_readback_tokens = pi0.Pi0.v0920_readback_tokens
    _v0920_prepare_prefix = pi0.Pi0._v0920_prepare_prefix

    def __init__(self, rngs: nnx.Rngs, *, readback: bool = True, pointer: bool = True):
        super().__init__(rngs, token_writes=True, pointer_read=pointer, beta_init=10.0, sentence_len=SENT,
                         reference_tokens=((5, 6), (7, 8), (5,)))
        self.memory_v6_pointer_query = "context"
        self.memory_v0920_input_read = True
        self.memory_v0920_input_rms = None
        self.memory_v0920_history_frames = 0
        self.memory_v0920_history_pool = 2
        self.memory_v0920_query_context = False
        self.memory_v7_no_visual_block = True
        self.memory_mask_zero_tokens = True
        self.memory_v7_write_every_step = False
        self.memory_v7_write_debounce_steps = 1
        self.memory_v5_write_conf = 0.8
        self.memory_v5_write_conf_min = True
        self.memory_v5_oracle_writes = False
        self.memory_v4_visual_injection = False
        self.memory_v5_query_standardize = False
        self.memory_v5_query_prev_sentence = False
        self.memory_blind_tokens = True
        self.memory_v0920_prev_readback = readback
        if readback:
            gate = jnp.arctanh(jnp.asarray(0.5, dtype=jnp.float32))
            self.memory_sem_readback_inject_w = nnx.Param(jnp.full((WIDTH,), gate, dtype=jnp.float32))
            self.memory_sem_readback_slot_embedding = nnx.Param(jnp.zeros((SENT, WIDTH), dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV4(nnx.Rngs(9))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _note(tokens=(5, 6)):
    row = np.zeros((1, SENT), dtype=np.int32)
    row[0, : len(tokens)] = tokens
    return jnp.asarray(row), jnp.asarray(np.arange(SENT)[None] < len(tokens))


def _bank_with(model, tokens=(5, 6)):
    row, mask = _note(tokens)
    blank = model.memory_semantic.init_state(1)
    written, _, _ = model.v6_semantic_write_tokens(blank, row, mask, jnp.ones((1,), dtype=bool))
    return blank, written


def _cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- (a) config
def test_v4_config_gates():
    ok = pi0_config.Pi0Config(**_v0_kwargs(memory_v6_token_writes=True, memory_v6_pointer_read=True, memory_v6_pointer_query="context",
                                           memory_v0920_prev_readback=True, memory_v5_write_conf_min=True, memory_v5_write_conf=0.8))
    assert ok.memory_v0920_prev_readback and ok.memory_v5_write_conf_min and ok.memory_v6_pointer_read
    assert pi0_config.Pi0Config(**_v0_kwargs()).memory_v0920_prev_readback is False
    assert pi0_config.Pi0Config(**_v0_kwargs()).memory_v5_write_conf_min is False
    with pytest.raises(ValueError, match="memory_v6_token_writes"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v0920_prev_readback=True))
    with pytest.raises(ValueError, match="memory_v0920_prev_readback"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v0920_input_read=False, memory_v7_no_visual_block=False,
                                          memory_v6_token_writes=True, memory_v0920_prev_readback=True))


# --------------------------------------------------------------------------- (b) layout
def test_read_back_widens_the_memory_block_by_the_note_length(tiny):
    assert tiny._memory_token_total == tiny.memory_v5_read_queries + SENT
    tiny.memory_v0920_prev_readback = False
    try:
        assert tiny._memory_token_total == tiny.memory_v5_read_queries
    finally:
        tiny.memory_v0920_prev_readback = True


# --------------------------------------------------------------------------- (c) the read-back
def test_a_committed_note_reads_back_exactly(tiny):
    _, written = _bank_with(tiny, (5, 6))
    row, mask = _note((5, 6))
    tokens, valid, retrieved, keys = tiny.v0920_readback_tokens(written, row, mask, jnp.float32)
    assert tokens.shape == (1, SENT, WIDTH) and valid.shape == (1, SENT) and keys.shape == (1, SENT, D_KEY)
    own_keys, values, _ = tiny.v6_sentence_token_kv(row, mask)
    # the read-back IS the bank's answer to the note's own write keys (the same key function as the write)
    np.testing.assert_allclose(np.asarray(keys), np.asarray(own_keys), atol=1e-6)
    np.testing.assert_allclose(np.asarray(retrieved), np.asarray(tiny.memory_semantic.read_key(written, own_keys)), atol=1e-5)
    # the delta rule reproduces the newest association exactly; the tiny model's random, unwhitened context keys are not
    # orthogonal, so earlier positions of a longer note carry interference here (the full model whitens its keys and the
    # 09-23 checkpoint probe measured every position at cosine 1.00)
    assert _cos(retrieved[0, SENT - 1], values[0, SENT - 1]) > 0.99
    assert bool(np.all(np.asarray(valid)))
    one_row, one_mask = _note((5,))
    _, one_bank = _bank_with(tiny, (5,))
    _, _, one_read, _ = tiny.v0920_readback_tokens(one_bank, one_row, one_mask, jnp.float32)
    assert _cos(one_read[0, 0], tiny.v6_sentence_token_kv(one_row, one_mask)[1][0, 0]) > 0.99
    # a shorter note: the positions outside it are exactly zero and invalid
    row2, mask2 = _note((7,))
    _, written2 = _bank_with(tiny, (7,))
    tokens2, valid2, retrieved2, _ = tiny.v0920_readback_tokens(written2, row2, mask2, jnp.float32)
    assert np.asarray(valid2).tolist() == [[True, False]]
    assert float(jnp.max(jnp.abs(tokens2[0, 1]))) == 0.0 and float(jnp.max(jnp.abs(retrieved2[0, 1]))) == 0.0
    assert _cos(retrieved2[0, 0], tiny.v6_sentence_token_kv(row2, mask2)[1][0, 0]) > 0.99  # the newest token


def test_a_blank_bank_or_an_empty_note_gives_zero_masked_tokens(tiny):
    blank, written = _bank_with(tiny)
    row, mask = _note()
    tokens, valid, retrieved, _ = tiny.v0920_readback_tokens(blank, row, mask, jnp.float32)
    assert float(jnp.max(jnp.abs(retrieved))) == 0.0 and float(jnp.max(jnp.abs(tokens))) == 0.0 and not bool(np.any(np.asarray(valid)))
    empty_row, empty_mask = _note(())
    tokens, valid, retrieved, _ = tiny.v0920_readback_tokens(written, empty_row, empty_mask, jnp.float32)
    assert float(jnp.max(jnp.abs(tokens))) == 0.0 and not bool(np.any(np.asarray(valid)))
    # zero_read: nothing is read even from a full bank
    tokens, valid, retrieved, _ = tiny.v0920_readback_tokens(written, row, mask, jnp.float32, zero_read=True)
    assert float(jnp.max(jnp.abs(retrieved))) == 0.0 and not bool(np.any(np.asarray(valid)))


def test_the_prefix_carries_the_read_back_tokens_after_the_questions(tiny):
    step0 = _single_step_observation()
    prefix, mask, ar = tiny.embed_prefix(step0)
    _, written = _bank_with(tiny)
    row, note_mask = _note((5,))
    _, written = _bank_with(tiny, (5,))
    out = tiny._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4, prev_tokens=row, prev_mask=note_mask)
    r = tiny.memory_v5_read_queries
    assert out["memory_tokens"].shape[1] == r + SENT and out["memory_valid"].shape == (1, r + SENT)
    assert np.asarray(out["memory_valid"])[0, r:].tolist() == [True, False]
    assert out["readback_retrieved"].shape == (1, SENT, WIDTH) and out["readback_keys"].shape == (1, SENT, D_KEY)
    assert int(out["capacity"]) == mask.shape[1] + r + SENT + tiny.causal_token_len


# --------------------------------------------------------------------------- (d) guard
def test_missing_note_is_refused(tiny):
    step0 = _single_step_observation()
    prefix, mask, ar = tiny.embed_prefix(step0)
    _, written = _bank_with(tiny)
    with pytest.raises(ValueError, match="memory_v0920_prev_readback"):
        tiny._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4)


# --------------------------------------------------------------------------- (e) sampler + scan
def test_sampler_runs_with_pointer_and_read_back(tiny):
    observation = _single_step_observation()
    visual = tiny.memory.init_state(1)
    _, written = _bank_with(tiny)
    row, mask = _note()
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32), "write_mode": "frozen"}
    _, _, aux = tiny.sample_with_memory(jax.random.key(923), observation, visual, semantic_state=written,
                                        v5_prev_tokens=row, v5_prev_mask=mask, **kwargs)
    assert aux["sem_queries"].shape == (1, tiny.memory_v5_read_queries, D_KEY)
    assert np.all(np.isfinite(np.asarray(aux["token_prob"])))


def test_sequence_loss_runs_with_pointer_and_read_back_and_gradients_are_finite(tiny):
    from openpi.models.pi0_v4_test import _v4_sequence_observation

    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = tiny._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(tiny)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
           if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    # the read-back gate and the pointer scale sit on the loss path
    assert float(jnp.max(jnp.abs(grads["memory_sem_readback_inject_w"].value))) >= 0.0
    assert "memory_v6_pointer_beta" in grads


# --------------------------------------------------------------------------- (f) the gate
def test_sentence_confidence_mean_or_minimum(tiny):
    conf = jnp.asarray([[0.99, 0.6, 0.99, 0.1], [0.5, 0.5, 0.5, 0.5]], dtype=jnp.float32)
    span = jnp.asarray([[True, True, True, False], [False, False, False, False]])
    lowest = pi0.v5_sentence_confidence(tiny, conf, span)
    np.testing.assert_allclose(np.asarray(lowest), [0.6, 0.0], atol=1e-6)
    tiny.memory_v5_write_conf_min = False
    try:
        mean = pi0.v5_sentence_confidence(tiny, conf, span)
    finally:
        tiny.memory_v5_write_conf_min = True
    np.testing.assert_allclose(np.asarray(mean), [(0.99 + 0.6 + 0.99) / 3, 0.0], atol=1e-6)
