"""v7 phase context transforms."""

import numpy as np
import pytest

import openpi.transforms as _transforms


def test_split_state_history():
    tf = _transforms.SplitStateHistory(history=2)
    out = tf({"state": np.arange(6).reshape(3, 2), "other": 1})
    assert np.array_equal(out["state"], [4, 5]) and np.array_equal(out["state_history"], [[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        tf({"state": np.zeros((2, 2))})


def test_prev_subtask_from_table():
    tasks = {0: "watch, sago left bin", 1: "first, get cup"}
    table = (np.array([0] * 20 + [1] * 20),)
    tf = _transforms.PrevSubtaskFromTable(table, tasks, stride=15)
    assert tf({"episode_index": np.int64(0), "frame_index": np.int64(30)})["prev_subtask"] == "watch, sago left bin"
    assert tf({"episode_index": np.int64(0), "frame_index": np.int64(39)})["prev_subtask"] == "first, get cup"
    assert tf({"episode_index": np.int64(0), "frame_index": np.int64(3)})["prev_subtask"] == "watch, sago left bin"  # clamped
    dropped = _transforms.PrevSubtaskFromTable(table, tasks, stride=15, dropout=1.0)
    assert dropped({"episode_index": np.int64(0), "frame_index": np.int64(30)})["prev_subtask"] == "none"
    # SubtaskFromLeRobotTask is untouched: a [T] sequence still yields a list, a scalar a string
    assert _transforms.SubtaskFromLeRobotTask(tasks)({"task_index": np.array([0, 1])})["subtask"] == list(tasks.values())
    assert _transforms.SubtaskFromLeRobotTask(tasks)({"task_index": np.array(1)})["subtask"] == "first, get cup"


def test_sidecar_prev_stride_clamps_to_episode_start():
    table = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    tf = _transforms.SubtaskFromV5Sidecar((table,), prev_stride=15)
    out = tf({"episode_index": np.int64(0), "frame_index": np.int64(12)})
    assert out["subtask"] == "b" and out["prev_subtask"] == "a"
    out0 = tf({"episode_index": np.int64(0), "frame_index": np.int64(3)})
    assert out0["prev_subtask"] == "a"


def test_tokenize_transform_pops_context_fields():
    class Tok:
        def tokenize(self, prompt, state, subtask, actions, *, state_history=None, prev_subtask=None):
            assert state_history is not None and prev_subtask == "first, get cup"
            return (np.zeros(4), np.ones(4, bool), np.zeros(4), np.zeros(4, bool), np.zeros(4, bool))

    tf = _transforms.TokenizeFASTSubtaskInputs(Tok())
    out = tf({"prompt": "p", "state": np.zeros(14), "subtask": "s", "state_history": np.zeros((1, 14)), "prev_subtask": np.asarray("first, get cup")})
    assert "state_history" not in out and "prev_subtask" not in out and "tokenized_prompt" in out
