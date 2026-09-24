"""Offline sentence evaluation must carry the auxiliary bank, not reset it each tick."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0_v0922ab_test import _tiny
from openpi.models.pi0_v35_test import _single_observation
from openpi.models import gemma
from v5_heldout_video import make_decode_fn


def test_offline_decode_requires_and_returns_carried_auxiliary_state(monkeypatch):
    monkeypatch.setattr(gemma, 'PALIGEMMA_VOCAB_SIZE', 128)
    model = _tiny(image=False, state=True)
    observation = _single_observation()
    sem = model.memory_semantic.init_state(1)
    visual = model.memory.init_state(1)
    previous = jnp.zeros((1, model.memory_v5_sentence_len), dtype=jnp.int32)
    mask = jnp.zeros_like(previous, dtype=bool)
    with pytest.raises(ValueError, match='carry_visual'):
        make_decode_fn(model, 2)(model, observation, None, sem, previous, mask)
    decode = make_decode_fn(model, 2, carry_visual=True)
    for tick in range(2):
        current = observation.replace(state=observation.state + tick + 1)
        keys, values, _ = model.vis_write_kv(jnp.zeros((1, 1, 64)), current.state)
        expected, _ = model.vis_bank_write(visual, keys, values, jnp.array([True]))
        *_, visual = decode(model, current, None, sem, previous, mask, visual)
        for a, b in zip(jax.tree.leaves(visual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
