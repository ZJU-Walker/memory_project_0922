"""v6 (cluster_v6/README.md): token-level contextual keys + pointer read on the tiny v5 stand-in.

Checks, in order: (a) the flags default off and the v5 sequence path is untouched; (b) causal token
states depend on the past only; (c) token-level writes store several same-shaped facts side by side and
read each back by its own context, while a rewrite of the same context replaces the value (newest wins);
(d) the pointer bonus is exactly zero at init (beta 0), lands only on reference tokens, and has finite
non-zero gradients; (e) the full sequence loss runs with both flags on and is finite with finite gradients.
"""

# ruff: noqa: SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v5_test import _TinyV5Seq
from openpi.models.pi0_v5_test import _v5_kwargs

WIDTH = 64
D_KEY = 32


class _TinyV6Seq(_TinyV5Seq):
    """The v5 tiny sequence model with the v6 methods and parameters."""

    v6_reference_token_ids = pi0.Pi0.v6_reference_token_ids
    _v6_reference_embed_stats = pi0.Pi0._v6_reference_embed_stats
    v6_token_values = pi0.Pi0.v6_token_values
    v6_sentence_token_kv = pi0.Pi0.v6_sentence_token_kv
    v6_semantic_write_tokens = pi0.Pi0.v6_semantic_write_tokens
    v6_pointer_bonus = pi0.Pi0.v6_pointer_bonus

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        token_writes: bool = True,
        pointer_read: bool = True,
        beta_init: float = 0.0,
        sentence_len: int = 4,
        reference_tokens: tuple[tuple[int, ...], ...] | None = None,
    ):
        super().__init__(rngs, oracle_writes=True)
        # a richer reference vocabulary: "<obj> in bin <k>"-shaped rows over a tiny token space
        # tokens: 10..13 objects, 20 = "in", 21 = "bin", 30..32 digits
        if reference_tokens is None:
            reference_tokens = tuple(
                (obj, 20, 21, digit) for obj in (10, 11, 12, 13) for digit in (30, 31, 32)
            ) + ((40, 41),)
        self.memory_v5_reference_tokens = reference_tokens
        self.memory_v5_sentence_len = sentence_len
        self.memory_v6_token_writes = token_writes
        self.memory_v6_pointer_read = pointer_read
        self.memory_v6_value_standardize = True
        # the v5 tiny bank (d_key 8, one 8-wide hidden layer) cannot hold four same-shaped facts apart; give
        # the v6 stand-in a bank with a realistic key/hidden ratio (real model: 512 / 1024)
        self.memory_semantic = memory.TitansMemory(
            memory.MemoryConfig(
                d_input=WIDTH,
                d_key=D_KEY,
                hidden_dims=(64,),
                d_value=WIDTH,
                mlp_l2norm=True,
                blank_initial_output=True,
                write_rule="delta_output",
                association_mode="pooled_frame",
                delta_rate=1.0,
                alpha_step=0.01,
            ),
            rngs=rngs,
        )
        self.memory_sem_key_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
        self.memory_sem_query_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
        if token_writes:
            self.memory_v6_token_key_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
            self.memory_v6_token_value_proj = nnx.Linear(WIDTH, WIDTH, use_bias=False, rngs=rngs)
            self.memory_v6_token_value_proj.kernel.value = jnp.eye(WIDTH, dtype=jnp.float32)
        if pointer_read:
            self.memory_v6_pointer_query_proj = nnx.Linear(WIDTH, D_KEY, use_bias=False, rngs=rngs)
            self.memory_v6_pointer_query_proj.kernel.value = self.memory_v6_pointer_query_proj.kernel.value * 0.02
            self.memory_v6_pointer_beta = nnx.Param(jnp.asarray(beta_init, dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny_v6():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV6Seq(nnx.Rngs(7))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


@pytest.fixture(scope="module")
def tiny_v6_seq():
    """Same model on the shared v4/v5 sequence fixture, whose causal buffer is 2 tokens wide."""
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV6Seq(nnx.Rngs(7), sentence_len=2, reference_tokens=((5, 6), (7, 8), (5,)))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _rows(*sentences: tuple[int, ...], length: int = 4):
    tokens = jnp.asarray([list(s) + [0] * (length - len(s)) for s in sentences], dtype=jnp.int32)
    mask = jnp.asarray([[True] * len(s) + [False] * (length - len(s)) for s in sentences], dtype=bool)
    return tokens, mask


# --------------------------------------------------------------------------- (a) config gating


def test_v6_flags_default_off_and_validate():
    cfg = pi0_config.Pi0Config(**_v5_kwargs(memory_v5_reference_tokens=((5, 6), (7, 8))))
    assert cfg.memory_v6_token_writes is False and cfg.memory_v6_pointer_read is False
    with pytest.raises(ValueError, match="need memory_v5_reference_tokens"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v6_token_writes=True))
    with pytest.raises(ValueError, match="replace the A8 slot keys"):
        pi0_config.Pi0Config(
            **_v5_kwargs(memory_v6_token_writes=True, memory_v5_slot_keys=True, memory_v5_reference_tokens=((5, 6),))
        )
    with pytest.raises(ValueError, match="needs memory_v6_token_writes"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v6_pointer_read=True, memory_v5_reference_tokens=((5, 6),)))
    with pytest.raises(ValueError, match="need memory_v5_sentence_bank"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_sentence_bank=False, memory_v6_token_writes=True))
    ok = pi0_config.Pi0Config(
        **_v5_kwargs(memory_v6_token_writes=True, memory_v6_pointer_read=True, memory_v5_reference_tokens=((5, 6),))
    )
    assert ok.memory_v6_pointer_beta_init == 0.0


# --------------------------------------------------------------------------- (b) causal states


def test_v6_causal_token_states_depend_on_the_past_only(tiny_v6):
    tokens, mask = _rows((10, 20, 21, 30), (10, 20, 21, 32))
    causal = np.asarray(tiny_v6._v5_token_states(tokens, mask, causal=True))
    bidir = np.asarray(tiny_v6._v5_token_states(tokens, mask, causal=False))
    # positions 0..2 share their past -> identical causal states; the last token differs
    np.testing.assert_allclose(causal[0, :3], causal[1, :3], atol=1e-5)
    assert not np.allclose(causal[0, 3], causal[1, 3], atol=1e-3)
    # the bidirectional pass lets the digit leak into the earlier positions
    assert not np.allclose(bidir[0, :3], bidir[1, :3], atol=1e-3)


# --------------------------------------------------------------------------- (c) token writes


def _read_digit(model, state, obj: int) -> tuple[int, float]:
    """Query the bank with the key that precedes the digit of '<obj> in bin ?' and decode among the digits."""
    tokens, mask = _rows((obj, 20, 21, 30))
    keys, _, _ = model.v6_sentence_token_kv(tokens, mask)
    query = keys[:, 3:4]  # the key of position 3 = context "<obj> in bin"
    read = np.asarray(model.memory_semantic.read_key(state, query))[0, 0]
    cand_tokens, cand_mask = _rows((30, 31, 32), length=3)
    cand_values = np.asarray(model.v6_token_values(cand_tokens, cand_mask))[0]
    scores = cand_values @ read / max(np.linalg.norm(read), 1e-9)
    order = np.argsort(-scores)
    return int(30 + order[0]), float(scores[order[0]] - scores[order[1]])


def test_v6_token_writes_store_four_facts_side_by_side_and_newest_wins(tiny_v6):
    model = tiny_v6
    state = model.memory_semantic.init_state(1)
    facts = {10: 30, 11: 32, 12: 31, 13: 32}
    for obj, digit in facts.items():
        tokens, mask = _rows((obj, 20, 21, digit))
        state, aux, pooled = model.v6_semantic_write_tokens(state, tokens, mask, jnp.asarray([True]))
        assert np.asarray(aux["commit_applied"]).shape == (1, 4)
        assert np.all(np.asarray(aux["commit_applied"]))
        np.testing.assert_allclose(np.linalg.norm(np.asarray(pooled)[0, 0]), 1.0, atol=1e-4)
    for obj, digit in facts.items():
        pred, margin = _read_digit(model, state, obj)
        assert pred == digit, (obj, pred, digit, margin)
        assert margin > 0.0
    # rewriting the same context with a new digit replaces the value for that object only
    tokens, mask = _rows((11, 20, 21, 30))
    state, _, _ = model.v6_semantic_write_tokens(state, tokens, mask, jnp.asarray([True]))
    assert _read_digit(model, state, 11)[0] == 30
    assert _read_digit(model, state, 10)[0] == 30 and _read_digit(model, state, 13)[0] == 32
    # a masked-off commit is exactly one decay step (nothing written)
    fresh = model.memory_semantic.init_state(1)
    decayed, aux, _ = model.v6_semantic_write_tokens(fresh, tokens, mask, jnp.asarray([False]))
    assert not np.any(np.asarray(aux["commit_applied"]))
    for name, leaf in decayed.fast_weights.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(fresh.fast_weights[name]))


# --------------------------------------------------------------------------- (d) pointer read


def test_v6_pointer_bonus_zero_at_init_reference_tokens_only_finite_grads(tiny_v6):
    model = tiny_v6
    state = model.memory_semantic.init_state(2)
    tokens, mask = _rows((10, 20, 21, 30), (12, 20, 21, 31))
    state, _, _ = model.v6_semantic_write_tokens(state, tokens, mask, jnp.asarray([True, True]))
    hidden = jax.random.normal(jax.random.key(1), (2, 4, WIDTH), dtype=jnp.float32)
    pos_mask = jnp.ones((2, 4), dtype=bool)
    bonus = model.v6_pointer_bonus(hidden, state, pos_mask, 128)
    assert bonus.shape == (2, 4, 128)
    np.testing.assert_array_equal(np.asarray(bonus), 0.0)  # beta = 0 at init
    model.memory_v6_pointer_beta.value = jnp.asarray(1.0, dtype=jnp.float32)
    try:
        bonus = np.asarray(model.v6_pointer_bonus(hidden, state, pos_mask, 128))
        ids = set(model.v6_reference_token_ids())
        others = [i for i in range(128) if i not in ids]
        np.testing.assert_array_equal(bonus[..., others], 0.0)
        assert np.any(bonus[..., sorted(ids)] != 0.0)
        # a masked position gets no bonus
        half = pos_mask.at[:, 2:].set(False)
        masked = np.asarray(model.v6_pointer_bonus(hidden, state, half, 128))
        np.testing.assert_array_equal(masked[:, 2:], 0.0)
    finally:
        model.memory_v6_pointer_beta.value = jnp.asarray(0.0, dtype=jnp.float32)
    # gradient reaches beta and the query map even from beta = 0
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(p):
        m = nnx.merge(graphdef, p, rest)
        b = m.v6_pointer_bonus(hidden, state, pos_mask, 128)
        return jnp.sum(jax.nn.log_softmax(b, axis=-1)[..., 30])

    grads = jax.grad(loss)(params)
    flat = {"/".join(str(k) for k in path): g for path, g in jax.tree_util.tree_leaves_with_path(grads)}
    beta_grad = [g for k, g in flat.items() if "memory_v6_pointer_beta" in k][0]
    assert np.isfinite(float(beta_grad)) and float(beta_grad) != 0.0
    for k, g in flat.items():
        assert np.all(np.isfinite(np.asarray(g))), k


# --------------------------------------------------------------------------- (e) full sequence


def test_v6_sequence_loss_is_finite_with_finite_gradients(tiny_v6_seq):
    model = tiny_v6_seq
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(46), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    # three sentence changes in the fixture -> three token-level commits
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    def total_loss(m):
        # training terms only (as in the v5 tests: the read-RMS telemetry has an infinite gradient at a blank bank)
        out = m._compute_sequence_loss_v32(jax.random.key(46), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(model)
    bad = [
        "/".join(str(k) for k in path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
    ]
    assert not bad, bad[:10]
    # the sentence CE reaches the v6 parameters: the pointer scale, the pointer query and the token projections
    assert float(jnp.abs(grads["memory_v6_pointer_beta"].value)) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_v6_token_key_proj"]["kernel"].value))) > 0.0


# --------------------------------------------------------------------------- (f) scan == unrolled slot loop


def test_v6_scan_slot_loop_matches_the_unrolled_loop(tiny_v6):
    """delta_write_kv_multi(slot_loop="scan") (v6 token writes, f = padded sentence length) is the unrolled
    per-slot loop of the v4/v5 callers: same state, same per-slot aux, masked slots included."""
    mem = tiny_v6.memory_semantic
    rng = np.random.default_rng(3)
    b, f = 2, 6
    k = jnp.asarray(rng.standard_normal((b, f, D_KEY)), dtype=jnp.float32)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jnp.asarray(rng.standard_normal((b, f, WIDTH)), dtype=jnp.float32)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    mask = jnp.asarray([[True, True, False, True, True, False], [True, False, True, True, False, False]])
    state = mem.init_state(b)
    s_unrolled, aux_unrolled = mem.delta_write_kv_multi(state, k, v, mask)
    s_scan, aux_scan = mem.delta_write_kv_multi(state, k, v, mask, slot_loop="scan")
    for name in s_unrolled.fast_weights:
        np.testing.assert_allclose(np.asarray(s_scan.fast_weights[name]), np.asarray(s_unrolled.fast_weights[name]), atol=1e-6, rtol=1e-6)
    assert np.array_equal(np.asarray(aux_scan["commit_applied"]), np.asarray(aux_unrolled["commit_applied"]))
    assert np.array_equal(np.asarray(aux_scan["commit_applied"]), np.asarray(mask))
    for name in ("pooled_key", "hidden", "pre_residual_norm", "surprise", "final_read_residual_norm"):
        np.testing.assert_allclose(np.asarray(aux_scan[name]), np.asarray(aux_unrolled[name]), atol=1e-6, rtol=1e-6)
    # a second write on the scanned state reads back the newest association (delta rule still exact)
    read = np.asarray(mem.read_key(s_scan, k[:, 4:5]))[:, 0]
    assert np.dot(read[0], np.asarray(v[0, 4])) > 0.9

