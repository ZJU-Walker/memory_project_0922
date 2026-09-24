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
