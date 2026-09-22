"""0920_v0 (robomme/docs/0920_v0_plan.md): fixed-query read at the INPUT, pooled front-camera history, write every tick.

On the tiny v6 stand-in (token-level writes, no pointer): (a) the config gate; (b) an empty bank reads exactly-zero,
masked tokens and a written bank reads finite tokens at the embedding scale, without the conditioner; (c) history keys
are pooled and time-stamped and a masked history frame changes nothing; (d) the full sequence loss runs with the flag,
is finite with finite gradients, reaches the read queries / slot embeddings / time embeddings and never the
conditioner; (e) write-every-tick commits at every step with a sentence; (f) the label-write probability 1 reproduces
the oracle-write bank and 0 does not; (g) the sampler runs and reads the bank.
"""

# ruff: noqa: SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v5_test import _v5_kwargs
from openpi.models.pi0_v6_test import _TinyV6Seq

WIDTH = 64
HISTORY = 2


def _v0_kwargs(**overrides) -> dict:
    values = _v5_kwargs(
        memory_v7_no_visual_block=True,
        memory_v0920_input_read=True,
        memory_v0920_history_frames=4,
        memory_v0920_history_pool=2,
        memory_v7_write_every_step=True,
        memory_v5_write_conf=0.0,
        memory_v5_reference_tokens=((5, 6), (7, 8)),
    )
    values.update(overrides)
    return values


def test_v0920_config_gate():
    config = pi0_config.Pi0Config(**_v0_kwargs())
    assert config.memory_v0920_input_read and config.memory_v0920_history_frames == 4
    assert pi0_config.Pi0Config(**_v5_kwargs()).memory_v0920_input_read is False
    with pytest.raises(ValueError, match="memory_v7_no_visual_block"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v7_no_visual_block=False))
    with pytest.raises(ValueError, match="pointer"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v6_pointer_read=True))
    with pytest.raises(ValueError, match="prompt slot"):
        pi0_config.Pi0Config(**_v0_kwargs(prompt_slot_len=16))
    with pytest.raises(ValueError, match="divide"):
        pi0_config.Pi0Config(**_v0_kwargs(memory_v0920_history_pool=3))


class _TinyV0(_TinyV6Seq):
    """The tiny v6 sequence model under the 0920_v0 flags: input read, no conditioner use, pooled history."""

    _top_camera_token_count = pi0.Pi0._top_camera_token_count
    _image_tower_tokens = pi0.Pi0._image_tower_tokens
    _augment_sequence_images = pi0.Pi0._augment_sequence_images
    _v0920_input_scale = pi0.Pi0._v0920_input_scale
    v0920_read_tokens = pi0.Pi0.v0920_read_tokens
    _v0920_prepare_prefix = pi0.Pi0._v0920_prepare_prefix

    def __init__(self, rngs: nnx.Rngs, *, history_frames: int = HISTORY):
        super().__init__(rngs, token_writes=True, pointer_read=False, sentence_len=2, reference_tokens=((5, 6), (7, 8), (5,)))
        self.memory_v0920_input_read = True
        self.memory_v0920_input_rms = None
        self.memory_v0920_history_frames = history_frames
        self.memory_v0920_history_pool = 2
        if history_frames > 0:
            self.memory_v0920_history_time = nnx.Param(jnp.zeros((history_frames, WIDTH), dtype=jnp.float32))
        self.memory_v7_no_visual_block = True
        self.memory_mask_zero_tokens = True
        self.memory_v7_write_every_step = True
        self.memory_v7_write_debounce_steps = 1
        self.memory_v5_write_conf = 0.0
        self.memory_v5_oracle_writes = False
        self.memory_v4_visual_injection = False
        self.memory_v5_query_standardize = False
        self.memory_v5_query_prev_sentence = False
        self.memory_blind_tokens = True


@pytest.fixture(scope="module")
def tiny_v0():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV0(nnx.Rngs(9))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _with_history(observation, *, frames: int = HISTORY, valid: bool = True):
    """Add `frames` front-camera history keys (scaled copies of the base camera) to a single-step or [b, t, ...]
    sequence observation."""
    base = observation.images["base_0_rgb"]
    base_mask = observation.image_masks["base_0_rgb"]
    images = dict(observation.images)
    masks = dict(observation.image_masks)
    for i in range(frames):
        images[f"history_{i}_rgb"] = base * (0.5 + 0.25 * i)
        masks[f"history_{i}_rgb"] = base_mask if valid else jnp.zeros_like(base_mask)
    return observation.replace(images=images, image_masks=masks)


def _single_step_observation():
    from openpi.models.pi0_v35_test import _single_observation

    return _single_observation()


def _written_bank(model, batch: int = 1):
    tokens = jnp.asarray([[5, 6]] * batch, dtype=jnp.int32)
    mask = jnp.ones((batch, 2), dtype=bool)
    blank = model.memory_semantic.init_state(batch)
    written, _ = model.v5_commit_sentence(blank, tokens, mask, jnp.ones((batch,), dtype=bool))
    return blank, written


# --------------------------------------------------------------------------- (b) the read


def test_v0920_read_is_zero_and_masked_on_an_empty_bank_and_scaled_after_a_write(tiny_v0):
    model = tiny_v0
    blank, written = _written_bank(model)
    tokens, valid, retrieved, queries, pre_rms, post_rms = model.v0920_read_tokens(blank, 1, jnp.float32)
    assert tokens.shape == (1, 3, WIDTH) and queries.shape == (1, 3, 32)
    np.testing.assert_array_equal(np.asarray(tokens), 0.0)  # slot embeddings are zero at init, the read is zero
    assert not bool(np.any(np.asarray(valid)))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(queries), axis=-1), 1.0, atol=1e-5)

    tokens_w, valid_w, retrieved_w, queries_w, pre_w, post_w = model.v0920_read_tokens(written, 1, jnp.float32)
    assert bool(np.all(np.asarray(valid_w)))
    assert np.all(np.isfinite(np.asarray(tokens_w)))
    np.testing.assert_array_equal(np.asarray(queries_w), np.asarray(queries))  # fixed queries: same on any bank
    # every token sits at gate * target RMS (tanh(w) = 0.5 at init, target = RMS of the reference-token embeddings)
    target = float(model._v0920_input_scale())
    per_token = np.sqrt(np.mean(np.square(np.asarray(tokens_w)), axis=-1))
    np.testing.assert_allclose(per_token, 0.5 * target, rtol=1e-4)
    assert float(pre_w[0]) > 0.0 and float(post_w[0]) > 0.0
    # a fixed target overrides the embedding scale
    model.memory_v0920_input_rms = 2.0
    try:
        tokens_f = model.v0920_read_tokens(written, 1, jnp.float32)[0]
        np.testing.assert_allclose(np.sqrt(np.mean(np.square(np.asarray(tokens_f)), axis=-1)), 1.0, rtol=1e-4)
    finally:
        model.memory_v0920_input_rms = None
    # zero_read: the tokens ignore the bank content
    tokens_z, valid_z = model.v0920_read_tokens(written, 1, jnp.float32, zero_read=True)[:2]
    np.testing.assert_array_equal(np.asarray(tokens_z), 0.0)
    assert not bool(np.any(np.asarray(valid_z)))


def test_v0920_prefix_pass_appends_the_memory_tokens_and_the_conditioner_is_unused(tiny_v0):
    model = tiny_v0
    step0 = _with_history(_single_step_observation())
    prefix, mask, ar = model.embed_prefix(step0)
    # 3 cameras x 4 tiny patch tokens + 2 history frames x 1 pooled token + 4 prompt tokens
    assert mask.shape[1] == 3 * 4 + HISTORY * 1 + 4
    assert model._top_camera_token_count(mask.shape[1] - 4, step0.images) == 4
    blank, written = _written_bank(model)
    prepared = model._v0920_prepare_prefix(prefix, mask, ar, written, top_token_count=4)
    assert prepared["memory_tokens"].shape == (1, 3, WIDTH)
    assert prepared["final_prefix"].shape == (1, mask.shape[1] + 3, WIDTH)
    assert prepared["cache"][0].shape[2] == mask.shape[1] + 3 + model.causal_token_len
    assert prepared["write_tokens"].shape == (1, model.memory_query_tokens, WIDTH)
    np.testing.assert_array_equal(np.asarray(prepared["write_tokens"]), 0.0)
    # the prefix rows depend on the bank content (memory columns are keys from block 0)
    prepared_blank = model._v0920_prepare_prefix(prefix, mask, ar, blank, top_token_count=4)
    assert not np.allclose(np.asarray(prepared_blank["final_prefix"][:, : mask.shape[1]]), np.asarray(prepared["final_prefix"][:, : mask.shape[1]]), atol=1e-6)

    def read_loss(m):
        p, mk, a = m.embed_prefix(step0)  # inside the graph: the history time embedding is added in embed_prefix
        return jnp.sum(m._v0920_prepare_prefix(p, mk, a, written, top_token_count=4)["final_prefix"])

    grads = nnx.grad(read_loss)(model)
    assert float(jnp.max(jnp.abs(grads["memory_sem_read_query_bank"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_slot_embedding"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_v0920_history_time"].value))) > 0.0
    cond = [float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grads["memory_sem_read_conditioner"])]
    assert max(cond) == 0.0


# --------------------------------------------------------------------------- (c) history


def test_v0920_history_frames_are_pooled_time_stamped_and_maskable(tiny_v0):
    model = tiny_v0
    plain = _single_step_observation()
    with_hist = _with_history(plain)
    masked_hist = _with_history(plain, valid=False)
    p_plain, m_plain, _ = model.embed_prefix(plain)
    p_hist, m_hist, _ = model.embed_prefix(with_hist)
    p_masked, m_masked, _ = model.embed_prefix(masked_hist)
    assert m_hist.shape[1] == m_plain.shape[1] + HISTORY
    # masked history frames are absent from the attention mask ...
    assert not bool(np.any(np.asarray(m_masked[:, 12 : 12 + HISTORY])))
    assert bool(np.all(np.asarray(m_hist[:, 12 : 12 + HISTORY])))
    # ... and a pooled history token is the mean of the 4 tiny patch tokens of that frame (time embedding zero at init)
    hist_tokens, _ = model.PaliGemma.img(with_hist.images["history_1_rgb"], train=False)
    np.testing.assert_allclose(np.asarray(p_hist[:, 13]), np.asarray(jnp.mean(hist_tokens, axis=1)), rtol=1e-5, atol=1e-6)
    # the time embedding is added per slot
    model.memory_v0920_history_time.value = model.memory_v0920_history_time.value.at[1].set(1.0)
    try:
        p_time, _, _ = model.embed_prefix(with_hist)
        np.testing.assert_allclose(np.asarray(p_time[:, 13] - p_hist[:, 13]), 1.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(p_time[:, 12]), np.asarray(p_hist[:, 12]), rtol=1e-6, atol=1e-7)
    finally:
        model.memory_v0920_history_time.value = jnp.zeros_like(model.memory_v0920_history_time.value)
    # a masked history frame leaves the prompt rows of the prefix pass identical to no history at all
    blank = model.memory_semantic.init_state(1)
    ar_plain = jnp.zeros_like(m_plain, dtype=jnp.int32)
    ar_masked = jnp.zeros_like(m_masked, dtype=jnp.int32)
    out_plain = model._v0920_prepare_prefix(p_plain, m_plain, ar_plain, blank, top_token_count=4)["final_prefix"]
    out_masked = model._v0920_prepare_prefix(p_masked, m_masked, ar_masked, blank, top_token_count=4)["final_prefix"]
    np.testing.assert_allclose(np.asarray(out_masked[:, :12]), np.asarray(out_plain[:, :12]), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_masked[:, 12 + HISTORY : 16 + HISTORY]), np.asarray(out_plain[:, 12:16]), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- (d)(e) the sequence loss


def test_v0920_sequence_loss_is_finite_writes_every_step_and_reaches_the_input_read(tiny_v0):
    model = tiny_v0
    observation = _with_history(_v4_sequence_observation())
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    # write every tick: all three steps carry a sentence -> three commits (the v6 fixture, changes only, commits 3 as
    # well; the repeated-sentence case is covered below)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    # the visual bank is out of the sequence: nothing is injected from it
    assert model._memory_token_total == 3

    def total_loss(m):
        out = m._compute_sequence_loss_v32(jax.random.key(920), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(model)
    bad = ["/".join(str(k) for k in path) for path, leaf in jax.tree_util.tree_leaves_with_path(grads) if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))]
    assert not bad, bad[:10]
    assert float(jnp.max(jnp.abs(grads["memory_sem_read_query_bank"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_slot_embedding"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_v0920_history_time"].value))) > 0.0
    cond = [float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(grads["memory_sem_read_conditioner"])]
    assert max(cond) == 0.0


def test_v0920_write_every_step_commits_repeated_sentences_and_keeps_the_bank_readable(tiny_v0):
    model = tiny_v0
    observation = _with_history(_v4_sequence_observation()).replace(
        tokenized_causal=jnp.asarray([[[5, 6], [5, 6], [5, 6]]], dtype=jnp.int32)
    )
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    model.memory_v5_oracle_writes = True
    try:
        losses = model._compute_sequence_loss_v32(jax.random.key(921), observation, actions, train=False)
    finally:
        model.memory_v5_oracle_writes = False
    # the same sentence three times: three commits under write-every-step (a changes-only rule would commit once)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    assert float(losses["v4_sem_raw_read_rms_sum"]) > 0.0


# --------------------------------------------------------------------------- (f) label-write schedule


def test_v0920_label_write_probability_one_matches_oracle_writes_and_zero_does_not(tiny_v0):
    model = tiny_v0
    observation = _with_history(_v4_sequence_observation())
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    keys = ("v4_decision_ce_steps", "v4_sem_commit_count", "v4_sem_raw_read_rms_sum")

    model.memory_v5_oracle_writes = True
    try:
        oracle = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    finally:
        model.memory_v5_oracle_writes = False
    p1 = model._compute_sequence_loss_v32(
        jax.random.key(922), observation.replace(seq_label_write_prob=jnp.ones((1,), dtype=jnp.float32)), actions, train=False
    )
    p0 = model._compute_sequence_loss_v32(
        jax.random.key(922), observation.replace(seq_label_write_prob=jnp.zeros((1,), dtype=jnp.float32)), actions, train=False
    )
    own = model._compute_sequence_loss_v32(jax.random.key(922), observation, actions, train=False)
    for key in keys:
        np.testing.assert_allclose(np.asarray(p1[key]), np.asarray(oracle[key]), rtol=1e-5, atol=1e-6, err_msg=key)
        np.testing.assert_allclose(np.asarray(p0[key]), np.asarray(own[key]), rtol=1e-5, atol=1e-6, err_msg=key)
    # the untrained decode is not the label: own writes give a different bank, hence a different read
    assert not np.allclose(np.asarray(p0["v4_sem_raw_read_rms_sum"]), np.asarray(p1["v4_sem_raw_read_rms_sum"]), rtol=1e-4)


# --------------------------------------------------------------------------- (g) the sampler


def test_v0920_sampler_runs_and_reads_the_bank(tiny_v0):
    from openpi.models.pi0_v35_test import _single_observation

    model = tiny_v0
    observation = _with_history(_single_observation())
    visual = model.memory.init_state(1)
    blank, written = _written_bank(model)
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32), "write_mode": "frozen"}
    _, state_blank, aux_blank = model.sample_with_memory(jax.random.key(923), observation, visual, semantic_state=blank, **kwargs)
    _, state_written, aux_written = model.sample_with_memory(jax.random.key(923), observation, visual, semantic_state=written, **kwargs)
    assert aux_blank["token_prob"].shape == (1, model.causal_token_len)
    assert aux_written["sem_queries"].shape == (1, 3, 32)
    jax.tree.map(np.testing.assert_array_equal, state_blank, visual)
    assert not (
        np.array_equal(np.asarray(aux_blank["tokens"]), np.asarray(aux_written["tokens"]))
        and np.allclose(np.asarray(aux_blank["token_prob"]), np.asarray(aux_written["token_prob"]))
    )


# --------------------------------------------------------------------------- (h) image tower outside the scan


def test_v0920_precomputed_image_tokens_match_the_in_scan_tower(tiny_v0):
    model = tiny_v0
    step0 = _with_history(_single_step_observation())
    ref = model.embed_prefix(step0)
    pre = {name: model._image_tower_tokens(name, img) for name, img in step0.images.items()}
    assert pre["history_0_rgb"].shape[1] == 1  # 4 tiny patch tokens pooled 2x2 -> 1
    got = model.embed_prefix(step0, image_tokens=pre)
    for a, b_ in zip(ref, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b_), rtol=1e-6, atol=1e-6)
    # the full sequence loss is the same with the tower hoisted out of the scan
    observation = _with_history(_v4_sequence_observation())
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    base = model._compute_sequence_loss_v32(jax.random.key(924), observation, actions, train=False)
    model.memory_v0920_vision_outside_scan = True
    try:
        hoisted = model._compute_sequence_loss_v32(jax.random.key(924), observation, actions, train=False)
    finally:
        model.memory_v0920_vision_outside_scan = False
    for key in ("v4_decision_ce_steps", "v4_sem_commit_count", "v4_sem_raw_read_rms_sum"):
        np.testing.assert_allclose(np.asarray(hoisted[key]), np.asarray(base[key]), rtol=1e-5, atol=1e-6, err_msg=key)
    # and the time embedding still receives a gradient through the hoisted path
    model.memory_v0920_vision_outside_scan = True
    try:
        def total_loss(m):
            out = m._compute_sequence_loss_v32(jax.random.key(924), observation, actions, train=False)
            return jnp.sum(out["v4_decision_ce_steps"])
        grads = nnx.grad(total_loss)(model)
    finally:
        model.memory_v0920_vision_outside_scan = False
    assert float(jnp.max(jnp.abs(grads["memory_v0920_history_time"].value))) > 0.0


# --------------------------------------------------------------------------- (i) decision ticks are supervised


def test_v0920_decision_ticks_keep_their_ce_and_flow_without_a_visual_commit(tiny_v0):
    """Bug found 09-21 00:55: the inert visual bank (zero write tokens) never 'commits', and the decision-tick losses
    were gated on that commit -> zero CE / flow on every count sentence. Under the v0 flag a valid transition suffices."""
    model = tiny_v0
    observation = _with_history(_v4_sequence_observation())
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = model._compute_sequence_loss_v32(jax.random.key(925), observation, actions, train=False)
    # the fixture's decision tick is step 2: no "state invalid" decision, and its CE is inside the training term
    np.testing.assert_array_equal(losses["v35_state_invalid_d_count"], 0.0)
    np.testing.assert_array_equal(losses["v35_commit_success_count"], 1.0)  # = the fixture's one write-mask step (no visual commit needed)
    no_decision = observation.replace(seq_decision_mask=jnp.zeros_like(observation.seq_decision_mask))
    losses_nd = model._compute_sequence_loss_v32(jax.random.key(925), no_decision, actions, train=False)
    # with the decision step removed from the decision mask nothing else changes, so the training CE must be identical:
    # under the old gating the decision step had been dropped from "ce" and the two would differ
    np.testing.assert_allclose(np.asarray(losses["ce"]), np.asarray(losses_nd["ce"]), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(losses["flow"]), np.asarray(losses_nd["flow"]), rtol=1e-6, atol=1e-7)
    assert float(jnp.sum(losses["ce"])) > 0.0


# --------------------------------------------------------------------------- (j) review gaps 09-21


def test_v0920_sampler_serves_a_request_without_the_blank_camera_and_uses_the_history(tiny_v0):
    """The served v0 request has base + wrist + 4 history keys and NO right wrist (RobommeInputs drops it): the sampler must
    preprocess the request's own keys, and a masked history frame must change the decode versus a visible one."""
    from openpi.models.pi0_v35_test import _single_observation

    model = tiny_v0
    base = _with_history(_single_observation())
    images = {k: v for k, v in base.images.items() if k != "right_wrist_0_rgb"}
    masks = {k: v for k, v in base.image_masks.items() if k != "right_wrist_0_rgb"}
    observation = base.replace(images=images, image_masks=masks)
    visual = model.memory.init_state(1)
    blank, written = _written_bank(model)
    kwargs = {"stop_token": 1, "max_decode_steps": 2, "num_steps": 1, "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32), "write_mode": "frozen"}
    _, _, aux = model.sample_with_memory(jax.random.key(926), observation, visual, semantic_state=written, **kwargs)
    assert aux["token_prob"].shape == (1, model.causal_token_len)
    hidden = observation.replace(image_masks={k: (jnp.zeros_like(v) if k.startswith("history_") else v) for k, v in masks.items()})
    _, _, aux_hidden = model.sample_with_memory(jax.random.key(926), hidden, visual, semantic_state=written, **kwargs)
    assert not (
        np.array_equal(np.asarray(aux["tokens"]), np.asarray(aux_hidden["tokens"]))
        and np.allclose(np.asarray(aux["token_prob"]), np.asarray(aux_hidden["token_prob"]))
    )


def test_v0920_history_dropout_masks_every_history_frame_of_a_sample_in_training(tiny_v0):
    """With the dropout drawn for a sample, its training loss equals the loss of the same sample with every history mask
    False (same augmentation key, so the pixels are identical); without the draw it does not."""
    model = tiny_v0
    observation = _with_history(_v4_sequence_observation())
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    keys = ("ce", "flow")
    model.memory_time_consistent_augmentation = True
    try:
        model.memory_v0920_history_dropout = 1.0 - 1e-6  # the draw hits every sample
        dropped = model._compute_sequence_loss_v32(jax.random.key(927), observation, actions, train=True)
        model.memory_v0920_history_dropout = 0.0
        masked_obs = observation.replace(
            image_masks={k: (jnp.zeros_like(v) if k.startswith("history_") else v) for k, v in observation.image_masks.items()}
        )
        masked = model._compute_sequence_loss_v32(jax.random.key(927), masked_obs, actions, train=True)
        kept = model._compute_sequence_loss_v32(jax.random.key(927), observation, actions, train=True)
    finally:
        model.memory_v0920_history_dropout = 0.0
    for key in keys:
        np.testing.assert_allclose(np.asarray(dropped[key]), np.asarray(masked[key]), rtol=1e-5, atol=1e-6, err_msg=key)
    assert not np.allclose(np.asarray(kept["ce"]), np.asarray(dropped["ce"]), rtol=1e-4)


def test_v0920_history_frames_share_the_front_camera_augmentation(tiny_v0):
    model = tiny_v0
    model.memory_time_consistent_augmentation = True
    observation = _with_history(_v4_sequence_observation())
    # make every history frame a copy of the front camera: after augmentation they must still be identical to it
    images = {k: (observation.images["base_0_rgb"] if k.startswith("history_") else v) for k, v in observation.images.items()}
    out = model._augment_sequence_images(jax.random.key(928), images)
    for k in images:
        if k.startswith("history_"):
            np.testing.assert_allclose(np.asarray(out[k]), np.asarray(out["base_0_rgb"]), rtol=1e-6, atol=1e-6)
    # ... whereas the wrist camera receives its own transform
    assert not np.allclose(np.asarray(out["left_wrist_0_rgb"]), np.asarray(out["base_0_rgb"]))
