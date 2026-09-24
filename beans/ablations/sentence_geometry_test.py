"""Diagnostic math and production-path coverage without a full checkpoint."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("sentence_geometry", Path(__file__).with_name("sentence_geometry.py"))
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def test_cosines_are_normalized_and_zero_safe():
    x = np.asarray([[2., 0.], [0., 3.], [0., 0.]])
    np.testing.assert_array_equal(geometry.cosine_matrix(x), np.diag([1., 1., 0.]))
    assert geometry.pair_summary(np.eye(2), np.zeros((2, 2), bool)) == {"pairs": 0}


@pytest.mark.parametrize("hidden", [(), (32, 32, 32)])
def test_probe_uses_template_groups_and_production_memory(hidden):
    from openpi.models import gemma
    from openpi.models.pi0_template_slot_test import TinySlot

    old = gemma.PALIGEMMA_VOCAB_SIZE
    gemma.PALIGEMMA_VOCAB_SIZE = 128
    try:
        model = TinySlot(hidden_dims=hidden)
        report = geometry.measure(model)
        assert report["template_groups"] == [0, 0, 1, 2, 3, 4]
        assert report["summary"]["write_key"]["same_template"]["min"] > .99999
        assert report["summary"]["template_hidden_feature"]["different_templates"]["pairs"] == 10
        assert min(report["single_write_recall_cosines"]) > .99999
        assert len(report["sequential_oracle_writes"]) == 10
        assert np.isfinite(np.asarray(report["matrices"]["whitened_write_value"])).all()
        # The temporary encoding cache is removed before returning to the caller.
        assert model.v5_encode_sentence.__self__ is model
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = old


def test_token_probe_scores_latest_contexts_and_reports_whole_sentence_interference():
    import flax.nnx as nnx
    from openpi.models import gemma
    from openpi.models.pi0_v6_test import _TinyV6Seq

    old = gemma.PALIGEMMA_VOCAB_SIZE
    gemma.PALIGEMMA_VOCAB_SIZE = 128
    try:
        model = _TinyV6Seq(nnx.Rngs(7), pointer_read=False, hidden_dims=(64, 64, 64), sentence_len=4,
                           reference_tokens=((10, 20, 21, 30), (10, 20, 21, 31), (11, 20, 21, 32), (40,)))
        model.memory_v6_whiten_keys = True
        report = geometry.measure(model)
        assert report["valid_token_count"] == 13
        assert report["context_count"] == 7  # 3 per object, plus singleton; first two positions alias
        assert min(report["single_write_recall_cosines"]) > .99999
        assert len(report["whole_sentence_write_recall"]) == 4
        assert len(report["sequential_oracle_writes"]) == 14
        for item in report["sequential_oracle_writes"]:
            assert np.isfinite(item["target_cosine"])
            if item["context_tokens"] == [10, 20, 21]:
                assert item["expected_token"] == (31 if item["order"] == "reference_order" else 30)
        assert np.isfinite(np.asarray(report["matrices"]["distinct_context_hidden_feature"])).all()
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = old
