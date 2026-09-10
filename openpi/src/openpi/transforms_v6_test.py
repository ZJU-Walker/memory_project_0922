"""v6.5 still-tail mask emitted by MemoryV34Labels (tail sentence is the CE target, arm not yet in the decision phase)."""

import numpy as np

from openpi import transforms


def test_memory_v34_labels_emits_still_tail_mask():
    t = transforms.MemoryV34Labels(
        subtask_vocab=("watching", "spoon in bin 2", "spoon in bin 2, go", "open bin 2"),
        evidence_subtasks=("spoon in bin 2",),
        memory_required_subtasks=("open bin 2",),
        tail_subtasks=("spoon in bin 2, go",),
    )
    # sidecar (CE target) says the tail from step 2; the dataset task (subtask_now) enters the decision phase at step 4
    data = {
        "subtask": ["watching", "spoon in bin 2", "spoon in bin 2, go", "spoon in bin 2, go", "spoon in bin 2, go", "spoon in bin 2, go"],
        "subtask_now": ["watching", "spoon in bin 2", "spoon in bin 2", "spoon in bin 2", "open bin 2", "open bin 2"],
    }
    out = t(dict(data))
    np.testing.assert_array_equal(out["seq_still_tail_mask"], [False, False, True, True, False, False])
    np.testing.assert_array_equal(out["seq_waiting_mask"], [False, False, False, False, True, True])
    # without tail sentences the field is absent (every older config)
    t0 = transforms.MemoryV34Labels(subtask_vocab=("a",), evidence_subtasks=(), memory_required_subtasks=("open bin 2",))
    assert "seq_still_tail_mask" not in t0(dict(data))
