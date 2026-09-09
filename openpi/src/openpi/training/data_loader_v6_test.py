"""v6.3 still-decision sampling boost (pure numpy helper of data_loader._sequence_sampling_info)."""

import numpy as np

from openpi.training import data_loader


def test_still_decision_boost_scales_only_covering_starts_and_renormalises():
    # one episode of 20 frames, stride 2, windows of up to 4 steps; the dataset decision starts at frame 12,
    # still frames = [8, 12)
    frame = np.arange(20)
    episode = np.zeros(20, dtype=np.int64)
    valid_steps = np.minimum((20 - frame + 1) // 2, 4).astype(np.int32)
    weights = np.ones(20) / 20
    out, before, after = data_loader._still_decision_boost(
        weights, frame=frame, episode=episode, valid_steps=valid_steps, stride=2,
        decision_lo=np.asarray([12]), still_frames=4, boost=4.0,
    )
    # a start f covers a still frame iff some f + 2k (k < 4) lies in [8, 12): f in 2..11
    covers = np.zeros(20, dtype=bool)
    for f in range(20):
        covers[f] = any(8 <= f + 2 * k < 12 for k in range(int(valid_steps[f])))
    assert covers.sum() == 10 and covers[2] and covers[11] and not covers[1] and not covers[12]
    np.testing.assert_allclose(out.sum(), 1.0)
    ratio = out / weights
    np.testing.assert_allclose(ratio[covers], ratio[covers][0])
    np.testing.assert_allclose(ratio[~covers], ratio[~covers][0])
    np.testing.assert_allclose(ratio[covers][0] / ratio[~covers][0], 4.0)
    assert 0 < before < after < 1
    # no decision in the episode -> untouched
    same, b0, a0 = data_loader._still_decision_boost(
        weights, frame=frame, episode=episode, valid_steps=valid_steps, stride=2,
        decision_lo=np.asarray([-1]), still_frames=4, boost=4.0,
    )
    np.testing.assert_allclose(same, weights)
    assert b0 == 0.0 and a0 == 0.0
