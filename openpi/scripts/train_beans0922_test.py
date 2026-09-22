"""beans0922 v3 telemetry: the sentence-quality keys are plain (not diagnostic/*), ratios of the batch sums, and finite on an
empty window. Run with PYTHONPATH=scripts."""

import jax.numpy as jnp
import pytest

import train


def _chunk(**overrides):
    base = {
        "write_valid_count": jnp.asarray([30.0, 10.0]),  # per-sample valid ticks -> 40 in the batch
        "v7_hard_token_count": jnp.asarray(8.0),
        "v5_exact_decision_sum": jnp.asarray(6.0),
        "v4_decision_count": jnp.asarray(12.0),
        "v5_exact_evidence_sum": jnp.asarray(27.0),
        "v5_evidence_count": jnp.asarray(30.0),
    }
    base.update(overrides)
    return base


def test_sentence_quality_keys_are_plain_ratios():
    info = train._sentence_quality_info(_chunk())
    assert set(info) == {"wrong_sentence_tokens_per_tick", "decision_sentence_exact", "evidence_sentence_exact"}
    assert not any(train._is_diagnostic_metric(k) for k in info)
    assert float(info["wrong_sentence_tokens_per_tick"]) == pytest.approx(0.2)
    assert float(info["decision_sentence_exact"]) == pytest.approx(0.5)
    assert float(info["evidence_sentence_exact"]) == pytest.approx(0.9)


def test_sentence_quality_keys_are_finite_on_an_empty_window():
    info = train._sentence_quality_info(_chunk(write_valid_count=jnp.zeros((2,)), v7_hard_token_count=jnp.asarray(0.0),
                                               v5_exact_decision_sum=jnp.asarray(0.0), v4_decision_count=jnp.asarray(0.0),
                                               v5_exact_evidence_sum=jnp.asarray(0.0), v5_evidence_count=jnp.asarray(0.0)))
    assert all(float(v) == 0.0 for v in info.values())
