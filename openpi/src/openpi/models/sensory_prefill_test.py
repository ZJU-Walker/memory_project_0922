"""Causal sensory replay: padding never writes/decays, valid past writes in order."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0
from openpi.models.pi0_v0922ab_test import _tiny
from openpi.models.pi0_v4_test import _v4_sequence_observation


def test_past_offsets_are_split_and_padding_is_masked():
    split = transforms.SplitSensoryPrefill(steps=3, stride=5)
    data = dict(frame_index=7, state=np.arange(10).reshape(5, 2), image=np.arange(5))
    result = split(data)
    np.testing.assert_array_equal(result['memory_vis_prefill_mask'], [False, False, True])
    np.testing.assert_array_equal(result['memory_vis_prefill_state'], data['state'][:3])
    np.testing.assert_array_equal(result['state'], data['state'][3:])
    np.testing.assert_array_equal(result['image'], [3, 4])
    with pytest.raises(ValueError, match='never truncate'):
        split(dict(data, frame_index=16))


def test_state_replay_matches_valid_past_writes_and_ignores_padding():
    model = _tiny(image=False, state=True)
    states = jnp.array([[[90., 90.], [1., 2.], [3., 4.]]])
    observation = _v4_sequence_observation().replace(
        memory_vis_prefill_mask=jnp.array([[False, True, True]]),
        memory_vis_prefill_state=states,
    )
    blank = model.memory.init_state(1)
    actual = pi0.Pi0.vis_prefill(model, observation, blank)
    expected = blank
    for i in (1, 2):
        keys, values, _ = model.vis_write_kv(jnp.zeros((1, 1, 64)), states[:, i])
        expected, _ = model.vis_bank_write(expected, keys, values, jnp.array([True]))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    empty = pi0.Pi0.vis_prefill(model, observation.replace(memory_vis_prefill_mask=jnp.zeros((1, 3), bool)), blank)
    for a, b in zip(jax.tree.leaves(empty), jax.tree.leaves(blank), strict=True):
        np.testing.assert_array_equal(a, b)
