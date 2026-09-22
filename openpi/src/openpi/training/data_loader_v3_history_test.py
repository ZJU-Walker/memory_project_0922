"""RoboMME v3 later-history emphasis (pure numpy helper of data_loader._sequence_sampling_info)."""

import numpy as np
import pytest

from openpi.training import data_loader


def _two_episodes():
    # episode 0: 12 frames, label changes at frames 4 and 8 (3 segments); episode 1: 6 frames, no change
    tasks0 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    tasks1 = np.array([5, 5, 5, 5, 5, 5])
    episode = np.concatenate([np.zeros(12, dtype=np.int64), np.ones(6, dtype=np.int64)])
    frame = np.concatenate([np.arange(12), np.arange(6)])
    return [tasks0, tasks1], episode, frame


def test_history_power_counts_changes_up_to_the_start_and_keeps_branch_masses():
    episode_tasks, episode, frame = _two_episodes()
    n = len(episode)
    weights = np.zeros(n)
    full = frame == 0
    slice_ok = (frame > 0) & (frame < 10) & (episode == 0)  # episode 0 frames 1..9
    mc_ok = (episode == 1) & (frame > 0)  # episode 1 frames 1..5
    weights[full] = 0.4 / full.sum()
    weights[slice_ok] = 0.3 / slice_ok.sum()
    weights[mc_ok] = 0.3 / mc_ok.sum()
    out, n_before, table = data_loader._history_power_reweight(
        weights, frame=frame, episode=episode, episode_tasks=episode_tasks, branches=(slice_ok, mc_ok), power=1.0
    )
    # changes counted up to and including the start frame: frame 4 already sits in the second segment
    np.testing.assert_array_equal(n_before[:12], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(n_before[12:], 0)
    np.testing.assert_allclose(out.sum(), 1.0)
    # branch masses unchanged, full starts untouched
    np.testing.assert_allclose(out[full], weights[full])
    np.testing.assert_allclose(out[slice_ok].sum(), 0.3)
    np.testing.assert_allclose(out[mc_ok].sum(), 0.3)
    # inside the slice branch: weight ratio 1 : 2 : 3 for starts after 0 / 1 / 2 changes
    w0, w1, w2 = out[1], out[4], out[8]
    np.testing.assert_allclose([w1 / w0, w2 / w0], [2.0, 3.0])
    # a branch whose starts all have the same count is renormalised back to itself
    np.testing.assert_allclose(out[mc_ok], weights[mc_ok])
    assert table[0][0] > table[0][1] and table[2][1] > table[2][0]


def test_history_power_zero_is_identity_and_power_two_squares():
    episode_tasks, episode, frame = _two_episodes()
    weights = np.ones(len(episode)) / len(episode)
    slice_ok = (episode == 0) & (frame > 0)
    same, _, _ = data_loader._history_power_reweight(
        weights, frame=frame, episode=episode, episode_tasks=episode_tasks, branches=(slice_ok,), power=0.0
    )
    np.testing.assert_allclose(same, weights)
    sq, _, _ = data_loader._history_power_reweight(
        weights, frame=frame, episode=episode, episode_tasks=episode_tasks, branches=(slice_ok,), power=2.0
    )
    np.testing.assert_allclose(sq[8] / sq[1], 9.0)
    np.testing.assert_allclose(sq[slice_ok].sum(), weights[slice_ok].sum())


def test_history_power_ignores_zero_weight_starts_and_empty_branches():
    episode_tasks, episode, frame = _two_episodes()
    weights = np.zeros(len(episode))
    weights[1] = 0.5
    weights[9] = 0.5
    slice_ok = (episode == 0) & (frame > 0)
    out, _, _ = data_loader._history_power_reweight(
        weights, frame=frame, episode=episode, episode_tasks=episode_tasks,
        branches=(slice_ok, np.zeros(len(episode), dtype=bool)), power=1.0,
    )
    assert out[5] == 0.0  # a start that had no mass gets none
    np.testing.assert_allclose(out[9] / out[1], 3.0)
    np.testing.assert_allclose(out.sum(), 1.0)


def test_data_config_rejects_negative_power():
    from openpi.training import config as _config

    with pytest.raises(ValueError, match="memory_start_history_power"):
        _config.DataConfig(memory_start_history_power=-1.0).__post_init__()
