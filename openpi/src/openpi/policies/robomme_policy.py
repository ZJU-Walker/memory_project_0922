"""RoboMME Panda I/O (cluster_robomme/README.md). The v7 model and memory architecture are unchanged.

State = 7 joint positions + the first finger width (m); actions = the 8 released ABSOLUTE `joint_action` values
(gripper -1 close / +1 open). The benchmark has a front and a wrist camera; the YAM policy's right wrist slot is
zero-filled with its mask false. No delta-action transform (RobommeDataConfig drops it).
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import yam_policy


@dataclasses.dataclass(frozen=True)
class RobommeInputs(yam_policy.YamInputs):
    # 0920_v0 (robomme/docs/0920_v0_plan.md A): `history_frames` past front-camera frames arrive as
    # "observation/history_<i>" (oldest first) with "observation/history_valid" ([H] per frame, [T, H] per sequence) and
    # become the image keys "history_<i>_rgb" with that mask; the blank right-wrist slot is dropped from the sequence
    # (its 256 padding tokens make room for the 4 x 64 history tokens).
    history_frames: int = 0
    drop_blank_camera: bool = False

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        state = np.asarray(data["observation/state"])
        if state.shape[-1] != 8:
            raise ValueError(f"RoboMME needs 7 joint positions + 1 finger width, got {state.shape}")
        if "actions" in data and np.asarray(data["actions"]).shape[-1] != 8:
            raise ValueError("RoboMME actions must be 7 absolute joint targets + gripper (-1 close, +1 open)")
        data["observation/right_wrist_image"] = np.zeros_like(data["observation/left_wrist_image"])
        result = super().__call__(data)
        result["image_mask"]["right_wrist_0_rgb"] = np.zeros_like(result["image_mask"]["right_wrist_0_rgb"], dtype=bool)
        if self.drop_blank_camera:
            del result["image"]["right_wrist_0_rgb"], result["image_mask"]["right_wrist_0_rgb"]
        if self.history_frames > 0:
            missing = [i for i in range(self.history_frames) if f"observation/history_{i}" not in data]
            if missing or "observation/history_valid" not in data:
                raise ValueError(f"RoboMME needs observation/history_0..{self.history_frames - 1} and observation/history_valid")
            valid = np.asarray(data["observation/history_valid"], dtype=bool)
            if valid.shape[-1] != self.history_frames:
                raise ValueError(f"observation/history_valid must have {self.history_frames} entries per step, got {valid.shape}")
            for i in range(self.history_frames):
                image = yam_policy._parse_image(data[f"observation/history_{i}"])  # noqa: SLF001
                if image.shape != result["image"]["base_0_rgb"].shape:
                    raise ValueError(f"observation/history_{i} {image.shape} must match the front image {result['image']['base_0_rgb'].shape}")
                result["image"][f"history_{i}_rgb"] = image
                result["image_mask"][f"history_{i}_rgb"] = valid[..., i] if valid.ndim == 2 else np.bool_(valid[i])
        return result


@dataclasses.dataclass(frozen=True)
class RobommeOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :8]}
