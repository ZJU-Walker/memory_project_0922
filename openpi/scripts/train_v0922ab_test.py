"""beans0922 ablation: the sensory-bank telemetry reaches W&B as plain keys (the recipe logs with log_diagnostics=False)."""

import numpy as np

import train as train_script

_KEYS = ("vis_valid_count", "vis_bank_norm_sum", "vis_raw_read_rms_sum", "vis_injected_pre_cast_rms_sum", "vis_commit_count")


def test_vis_bank_info_is_the_per_valid_tick_mean_and_not_diagnostic():
    chunked = dict(zip(_KEYS, map(np.float32, (8.0, 16.0, 4.0, 2.0, 8.0))))
    info = train_script._vis_bank_info(chunked)
    assert set(info) == {"vis_bank_norm", "vis_read_rms", "vis_read_injected_rms", "vis_commit_rate"}
    assert not any(train_script._is_diagnostic_metric(k) for k in info)
    got = [float(info[k]) for k in ("vis_bank_norm", "vis_read_rms", "vis_read_injected_rms", "vis_commit_rate")]
    np.testing.assert_allclose(got, [2.0, 0.5, 0.25, 1.0])


def test_vis_bank_info_empty_window_is_exactly_zero():
    info = train_script._vis_bank_info({k: np.float32(0.0) for k in _KEYS})
    assert all(float(v) == 0.0 for v in info.values())


def test_the_raw_sums_and_the_denominator_stay_under_diagnostic():
    assert set(_KEYS) <= set(train_script._V5_INFO_KEYS)
    assert all(train_script._is_diagnostic_metric(f"diagnostic/{k}") for k in _KEYS)
