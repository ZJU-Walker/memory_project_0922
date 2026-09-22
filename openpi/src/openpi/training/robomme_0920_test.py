"""0920_v0 data path on CPU without a dataset: the loader-side history split, the RoboMME input mapping (history keys in,
blank camera out), the eval request decoder and the client ring buffer (robomme/docs/0920_v0_plan.md A, B, D)."""

import importlib.util
import pathlib

import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import robomme_policy


def _load_common():
    path = pathlib.Path(__file__).resolve().parents[3] / "cluster_robomme" / "eval" / "common.py"
    spec = importlib.util.spec_from_file_location("robomme_eval_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_image_history_restores_current_frames_and_flags_padding():
    steps, frames, per = 3, 4, 5
    # frame index encoded in the pixel value: tick k fetches offsets (-16, -12, -8, -4, 0) around frame 20 * k
    grid = np.stack([np.full((3, 4, 4), max(0, 20 * k + o), dtype=np.float32) / 255.0 for k in range(steps) for o in (-16, -12, -8, -4, 0)])
    is_pad = np.array([20 * k + o < 0 for k in range(steps) for o in (-16, -12, -8, -4, 0)])
    out = transforms.SplitImageHistory(key="image", frames=frames)({"image": grid, "image_is_pad": is_pad, "state": np.zeros((steps, 8))})
    assert out["image"].shape == (steps, 3, 4, 4) and "image_is_pad" not in out
    np.testing.assert_allclose(out["image"][:, 0, 0, 0] * 255, [0, 20, 40])
    np.testing.assert_allclose(out["history_0"][:, 0, 0, 0] * 255, [0, 4, 24])  # t-16 clamped to frame 0 at tick 0
    np.testing.assert_allclose(out["history_3"][:, 0, 0, 0] * 255, [0, 16, 36])
    np.testing.assert_array_equal(out["history_valid"], [[False] * 4, [True] * 4, [True] * 4])
    with pytest.raises(ValueError, match="T x 5"):
        transforms.SplitImageHistory(key="image", frames=frames)({"image": grid[:-1], "image_is_pad": is_pad[:-1]})


def _raw_sequence(steps=2, frames=4, size=8):
    rng = np.random.default_rng(0)
    data = {
        "observation/image": rng.integers(256, size=(steps, size, size, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(256, size=(steps, size, size, 3), dtype=np.uint8),
        "observation/state": rng.random((steps, 8), dtype=np.float32),
        "actions": rng.random((steps, 4, 8), dtype=np.float32),
        "observation/history_valid": np.array([[False, False, True, True], [True] * 4]),
    }
    for i in range(frames):
        data[f"observation/history_{i}"] = rng.integers(256, size=(steps, size, size, 3), dtype=np.uint8)
    return data


def test_robomme_inputs_map_history_keys_and_drop_the_blank_camera():
    data = _raw_sequence()
    out = robomme_policy.RobommeInputs(model_type=_model.ModelType.PI05, history_frames=4, drop_blank_camera=True)(dict(data))
    assert sorted(out["image"]) == ["base_0_rgb", "history_0_rgb", "history_1_rgb", "history_2_rgb", "history_3_rgb", "left_wrist_0_rgb"]
    for i in range(4):
        np.testing.assert_array_equal(out["image"][f"history_{i}_rgb"], data[f"observation/history_{i}"])
        np.testing.assert_array_equal(out["image_mask"][f"history_{i}_rgb"], data["observation/history_valid"][:, i])
    assert out["image_mask"]["base_0_rgb"].shape == (2,) and out["image_mask"]["base_0_rgb"].all()
    # legacy call (no history): the three-camera layout with the blank right wrist masked, bit for bit as before
    legacy = robomme_policy.RobommeInputs(model_type=_model.ModelType.PI05)({k: v for k, v in data.items() if "history" not in k})
    assert sorted(legacy["image"]) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert not legacy["image_mask"]["right_wrist_0_rgb"].any()
    with pytest.raises(ValueError, match="history_0"):
        robomme_policy.RobommeInputs(model_type=_model.ModelType.PI05, history_frames=4)({k: v for k, v in data.items() if "history" not in k})
    # single-frame serving request: scalar masks
    single = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim >= 1 and k != "observation/history_valid" else v) for k, v in data.items()}
    single["observation/history_valid"] = np.array([False, True, True, True])
    del single["actions"]
    out1 = robomme_policy.RobommeInputs(model_type=_model.ModelType.PI05, history_frames=4, drop_blank_camera=True)(single)
    assert out1["image"]["history_0_rgb"].shape == (8, 8, 3)
    assert bool(out1["image_mask"]["history_0_rgb"]) is False and bool(out1["image_mask"]["history_1_rgb"]) is True


def test_eval_request_decoder_accepts_the_history_keys_and_the_client_builds_them():
    common = _load_common()
    rng = np.random.default_rng(1)
    obs = {"prompt": "pick up the red cube two times", "observation/state": rng.random(8, dtype=np.float32),
           "observation/image": rng.integers(256, size=(6, 6, 3), dtype=np.uint8),
           "observation/left_wrist_image": rng.integers(256, size=(6, 6, 3), dtype=np.uint8)}
    for i in range(4):
        obs[f"observation/history_{i}"] = rng.integers(256, size=(6, 6, 3), dtype=np.uint8)
    obs["observation/history_valid"] = np.array([False, False, True, True])
    back = common.decode_observation(common.encode_observation(obs))
    for key, value in obs.items():
        np.testing.assert_array_equal(back[key], value)
    bad = dict(obs); del bad["observation/history_valid"]
    with pytest.raises(ValueError, match="history_valid"):
        common.decode_observation(common.encode_observation(bad))
    legacy = {k: v for k, v in obs.items() if "history" not in k}
    assert set(common.decode_observation(common.encode_observation(legacy))) == set(legacy)

    # the client ring buffer: observe() on every frame, history = t-16, t-12, t-8, t-4 with the clamp before frame 0
    client = common.HttpPolicyClient.__new__(common.HttpPolicyClient)
    client.state_history_steps, client.image_history_frames, client.image_history_stride = 0, 4, 4
    from collections import deque
    client._states, client._front = deque(maxlen=1), deque(maxlen=17)
    frames = [np.full((2, 2, 3), f, dtype=np.uint8) for f in range(30)]
    def raw(f):
        return {"front_rgb_list": [frames[f]], "wrist_rgb_list": [frames[f]], "joint_state_list": [np.zeros(7)], "gripper_state_list": [np.zeros(2)]}
    for f in range(30):
        client.observe(raw(f))
        if f in (0, 5, 20, 29):
            req = client._with_history(common.observation(raw(f), "goal"))
            got = [int(req[f"observation/history_{i}"][0, 0, 0]) for i in range(4)]
            expected = [max(0, f - back) for back in (16, 12, 8, 4)]
            assert got == expected, (f, got, expected)
            np.testing.assert_array_equal(req["observation/history_valid"], [f - back >= 0 for back in (16, 12, 8, 4)])
    with pytest.raises(ValueError, match="EVERY simulator frame"):
        client._with_history(common.observation(raw(3), "goal"))


def test_history_prefill_gaps_shift_under_write_every_step():
    """Write-every-tick keeps the current note at full strength; the prefill must decay note i only after note i+1 starts."""
    kwargs = dict(stride=20, steps=40, lookahead=0, episode_tasks=(), tasks={}, prefill_history=True, prefill_max=4)
    labels = ["a", "a", "a", "b", "b", "c"]
    plain = transforms.MemorySequenceSubtasks(**kwargs)._history_prefill(labels)
    every = transforms.MemorySequenceSubtasks(**kwargs, write_every_step=True)._history_prefill(labels)
    assert plain[0] == ["a", "b", "c", ""] and every[0] == ["a", "b", "c", ""]
    np.testing.assert_array_equal(plain[1], [2, 1, 0, 0])  # change-triggered: a decays over its own 3 ticks (2 gaps + b's commit)
    np.testing.assert_array_equal(every[1], [1, 0, 0, 0])  # every tick: a decays only after b starts, c not at all


def test_binfill_sets_are_label_derived():
    """Evidence = the sentences that open an object cycle; decision = those plus the stop. Nothing else, no ordering games."""
    from openpi.training import config as _config  # noqa: F401  (the config registry must be imported first: it imports the task modules)
    from openpi.training import robomme_0920_config as c

    sentences = ["all tasks completed", "pick up the first red cube", "pick up the second red cube", "press the button",
                 "put it into the bin"]
    evidence, decision = c.binfill_sets(sentences)
    assert evidence == ("pick up the first red cube", "pick up the second red cube")
    assert decision == evidence + ("press the button",)


def test_binfill_config_keeps_the_v0_recipe_with_its_own_window():
    """The BinFill family is the 0920_v0 recipe on another dataset: same model flags, tick 25 / 42-tick windows, the task's
    own paths and sentence sets; the PickXtimes config is untouched. Skipped until a BinFill spec is prepared."""
    from openpi.training import config as _config
    from openpi.training import robomme_0920_config as c

    root = pathlib.Path(__file__).resolve().parents[4]
    if not (root / "robomme/metadata" / c.BINFILL_DATASET / "prepared_official.json").is_file():
        pytest.skip(f"no prepared BinFill spec ({c.BINFILL_DATASET})")
    pick = _config.get_config("pi05_robomme_0920_v0")
    bin_ = _config.get_config("pi05_robomme_0920_binfill_v0")
    for flag in ("memory_v0920_input_read", "memory_v7_write_every_step", "memory_v5_write_conf", "memory_v0920_history_frames",
                 "memory_v0920_history_pool", "memory_v0920_history_dropout", "memory_v6_pointer_read", "prompt_slot_len",
                 "max_token_len", "action_horizon", "memory_v7_onset_ce_weight", "memory_state_mask_prob"):
        assert getattr(bin_.model, flag) == getattr(pick.model, flag), flag
    assert (pick.model.memory_seq_steps, pick.model.memory_block_steps) == (40, 20)
    assert (bin_.model.memory_seq_steps, bin_.model.memory_block_steps) == (c.BINFILL_SEQ_STEPS, c.BINFILL_BLOCK_STEPS) == (42, 21)
    dp, db = pick.data.create(pick.assets_dirs, pick.model), bin_.data.create(bin_.assets_dirs, bin_.model)
    assert (dp.memory_stride_frames, dp.memory_sequence_buckets, dp.memory_critical_start_pad) == (20, (10, 20, 30, 40), 100)
    assert (db.memory_stride_frames, db.memory_sequence_buckets, db.memory_critical_start_pad) == (25, (12, 22, 32, 42), 125)
    assert db.memory_image_history_frames == dp.memory_image_history_frames == 4 and db.memory_image_history_stride == 4
    assert db.repo_id == f"robomme/{c.BINFILL_DATASET}" and c.BINFILL_DATASET in db.lerobot_dataset_root
    assert db.memory_v5_subtask_labels_path.endswith(f"{c.BINFILL_DATASET}/subtasks_official.json")
    assert db.memory_episode_manifest_path.endswith(f"{c.BINFILL_DATASET}/manifest.json")
    assert bin_.data.assets.asset_id == c.BINFILL_DATASET
    assert set(db.evidence_subtasks) == {s for s in db.memory_subtask_vocab if s.startswith("pick up the")}
    assert set(db.memory_required_subtasks) == set(db.evidence_subtasks) | {"press the button"}
    assert all(s == s.lower() for s in db.memory_subtask_vocab)
    assert bin_.fsdp_devices == 2 and bin_.label_write_schedule_steps == pick.label_write_schedule_steps == 1_500
    assert bin_.num_train_steps == pick.num_train_steps == 5_001 and bin_.project_name == "robomme_0920"


def test_v1_is_v0_with_the_two_current_cameras_only():
    """v1 (09-21): the v0 recipe with no past frames, no blank slot, horizon 30. Model and data agree on 0 history frames, the
    RoboMME input mapping keeps front + wrist only (no history keys, no right-wrist slot), the plans are 30 steps, the weights
    come from the same plain pi05 base, and every other model / data / training field is bit-identical to v0 (batch is the
    v1 knob). v0 itself keeps its 4 frames. Same for the BinFill pair when its spec exists."""
    import dataclasses

    from openpi.training import config as _config
    from openpi.training import robomme_0920_config as c

    root = pathlib.Path(__file__).resolve().parents[4]
    pairs = [("pi05_robomme_0920_v0", "pi05_robomme_0920_v1"), ("pi05_robomme_0920_v0_smoke", "pi05_robomme_0920_v1_smoke")]
    if (root / "robomme/metadata" / c.BINFILL_DATASET / "prepared_official.json").is_file():
        pairs.append(("pi05_robomme_0920_binfill_v0", "pi05_robomme_0920_binfill_v1"))
    v1_model = {"memory_v0920_history_frames", "memory_v0920_history_dropout", "memory_v0920_drop_blank_camera", "action_horizon"}
    v1_data = {"memory_image_history_frames"}
    v1_train = {"name", "model", "data", "exp_name", "batch_size"}
    for name0, name1 in pairs:
        v0, v1 = _config.get_config(name0), _config.get_config(name1)
        assert v0.model.memory_v0920_history_frames == 4 and v1.model.memory_v0920_history_frames == 0, (name0, name1)
        assert v1.model.memory_v0920_history_dropout == 0.0 and v1.model.memory_v0920_drop_blank_camera
        assert (v0.model.action_horizon, v1.model.action_horizon) == (40, c.V1_ACTION_HORIZON) == (40, 30)
        for f in dataclasses.fields(v0.model):
            if f.name not in v1_model:
                assert getattr(v0.model, f.name) == getattr(v1.model, f.name), (name1, f.name)
        for f in dataclasses.fields(v0.data.base_config):
            if f.name not in v1_data:
                assert getattr(v0.data.base_config, f.name) == getattr(v1.data.base_config, f.name), (name1, f.name)
        for f in dataclasses.fields(v0):
            if f.name not in v1_train:
                assert getattr(v0, f.name) == getattr(v1, f.name), (name1, f.name)
        assert v1.batch_size == (c.V1_BATCH if "binfill" not in name1 else v0.batch_size)
        assert v1.weight_loader.params_path == v0.weight_loader.params_path == c.PI05_BASE_PARAMS  # the plain pi05 base
        d0, d1 = v0.data.create(v0.assets_dirs, v0.model), v1.data.create(v1.assets_dirs, v1.model)
        assert d0.memory_image_history_frames == 4 and d1.memory_image_history_frames == 0
        assert not any("history" in k for k in d1.repack_transforms.inputs[0].structure)
        assert any("history" in k for k in d0.repack_transforms.inputs[0].structure)
        r1 = [tr for tr in d1.data_transforms.inputs if isinstance(tr, robomme_policy.RobommeInputs)][0]
        assert r1.history_frames == 0 and r1.drop_blank_camera
        spec_obs, spec_act = v1.model.inputs_spec(batch_size=1)
        assert sorted(spec_obs.images) == ["base_0_rgb", "left_wrist_0_rgb"]
        assert spec_act.shape[-2] == 30
        spec0, _ = v0.model.inputs_spec(batch_size=1)
        assert sorted(spec0.images) == ["base_0_rgb", "history_0_rgb", "history_1_rgb", "history_2_rgb", "history_3_rgb", "left_wrist_0_rgb"]
    with pytest.raises(ValueError, match="0 .v1."):
        c._history_overrides(2)  # noqa: SLF001

