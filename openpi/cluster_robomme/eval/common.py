"""Observation-only transport and rollout contracts; no model or simulator imports."""
import base64
from collections import deque
import json
from urllib.request import Request, urlopen

import numpy as np

TASKS = ("BinFill", "PickXtimes", "SwingXtimes", "StopCube")


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def observation(obs, goal, index=-1):
    """Explicit allowlist: never transmit info/subgoal/privileged simulator state."""
    joint = array(obs["joint_state_list"][index]).reshape(-1)
    gripper = array(obs["gripper_state_list"][index]).reshape(-1)
    if joint.shape != (7,) or gripper.size != 2:
        raise ValueError("Expected Panda joint[7] and finger-width[2] observations")
    state = np.concatenate([joint, gripper[:1]]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite robot state")
    result = {"prompt": str(goal), "observation/state": state}
    for source, dest in (("front_rgb_list", "observation/image"),
                         ("wrist_rgb_list", "observation/left_wrist_image")):
        rgb = array(obs[source][index])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected HWC uint8 RGB camera observation")
        result[dest] = rgb
    return result


def validate_prediction(prediction, chunk_size):
    actions = np.asarray(prediction["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < chunk_size:
        raise ValueError("Expected at least {} absolute 8-D actions, got {}".format(chunk_size, actions.shape))
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite actions")
    if not isinstance(prediction.get("subtask"), str) and not (
        prediction.get("subtask") is None and prediction.get("subtask_source") == "not_predicted"
    ):
        raise ValueError("Policy must return its decoded subtask sentence or explicitly have no subtask predictor")
    return actions[:chunk_size]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def encode_observation(obs):
    return {key: ({"array": base64.b64encode(np.ascontiguousarray(val).tobytes()).decode(),
                   "dtype": str(val.dtype), "shape": list(val.shape)}
                  if isinstance(val, np.ndarray) else val) for key, val in obs.items()}


def decode_observation(data):
    keys = {"prompt", "observation/state", "observation/image", "observation/left_wrist_image"}
    if "observation/state_history" in data:
        keys.add("observation/state_history")
    # 0920_v0 short visual history: past FRONT frames observation/history_<i> (0..H-1, oldest first, the robot's own
    # camera stream only) + observation/history_valid[H] (False for frames before the episode start)
    history = sorted(k for k in data if k.startswith("observation/history_") and k != "observation/history_valid")
    if history:
        if history != ["observation/history_{}".format(i) for i in range(len(history))] or "observation/history_valid" not in data:
            raise ValueError("Image history must be observation/history_0..H-1 plus observation/history_valid")
        keys.update(history)
        keys.add("observation/history_valid")
    if set(data) != keys or not isinstance(data["prompt"], str):
        raise ValueError("Only RGB, joint/finger state and task instruction are accepted")
    result = {"prompt": data["prompt"]}
    for key in keys - {"prompt"}:
        val = data[key]
        dtype = {"float32": np.float32, "uint8": np.uint8, "bool": np.bool_}[val["dtype"]]
        result[key] = np.frombuffer(base64.b64decode(val["array"], validate=True), dtype=dtype).reshape(val["shape"])
        if key.endswith("state_history"):
            if result[key].shape != (2, 8) or dtype != np.float32 or not np.isfinite(result[key]).all():
                raise ValueError("Expected finite float32 state_history[2,8]")
        elif key.endswith("history_valid"):
            if result[key].shape != (len(history),) or dtype != np.bool_:
                raise ValueError("Expected bool history_valid[H]")
        elif key.endswith("state"):
            if result[key].shape != (8,) or dtype != np.float32 or not np.isfinite(result[key]).all():
                raise ValueError("Expected finite float32 state[8]")
        elif result[key].ndim != 3 or result[key].shape[-1] != 3 or dtype != np.uint8:
            raise ValueError("Expected HWC uint8 RGB")
    if history and any(result[k].shape != result["observation/image"].shape for k in history):
        raise ValueError("History frames must match the front image shape")
    return result


def decode_policy_request(data, allow_oracle=False):
    """Oracle labels require an explicitly diagnostic server; normal input stays strict."""
    payload = dict(data)
    forced = payload.pop("diagnostic_subtask", None) if allow_oracle else None
    if allow_oracle and (not isinstance(forced, str) or not forced.strip() or len(forced) > 2000):
        raise ValueError("Oracle diagnostic requires a nonempty official online subtask")
    return decode_observation(payload), forced


class HttpPolicyClient:
    """Separate simulator/model environments, using only stdlib HTTP and NumPy."""
    def __init__(self, url, task, chunk_size=5, timeout=1800, *, execution_diagnostic=False):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.metadata = self.request("/metadata")
        self.state_history_steps = int(self.metadata.get("prompt_state_history", 0))
        self.state_history_stride = int(self.metadata.get("prompt_state_history_stride", 10))
        self._states = deque(maxlen=self.state_history_steps * self.state_history_stride + 1)
        # 0920_v0: past front frames (client-side ring buffer, filled by observe() on EVERY simulator frame)
        self.image_history_frames = int(self.metadata.get("image_history_frames", 0))
        self.image_history_stride = int(self.metadata.get("image_history_stride", 4))
        self._front = deque(maxlen=self.image_history_frames * self.image_history_stride + 1)
        # diagnostic (09-21): send the history frames but mark every slot invalid, i.e. the 20 % history-dropout
        # condition of training; set by rollout.py --mask-history. Not benchmark eligible.
        self.mask_history = False
        if self.metadata.get("task") != task or self.metadata.get("action_space") != "joint_angle":
            raise ValueError("Server task/action contract does not match requested evaluation")
        trained_chunk = self.metadata.get("execution_chunk", self.metadata.get("memory_stride_frames"))
        if trained_chunk != chunk_size and not execution_diagnostic:
            raise ValueError("Execution chunk must equal the checkpoint memory stride")
        if execution_diagnostic:
            self.metadata = dict(self.metadata, benchmark_eligible=False, diagnostic_action_chunk=True,
                                 default_execution_chunk=trained_chunk, execution_chunk=chunk_size,
                                 inference_memory_stride_frames=chunk_size, action_offset=0)

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    @property
    def needs_observe(self):
        return bool(self.state_history_steps or self.image_history_frames)

    def reset(self, seed):
        self._states.clear()
        self._front.clear()
        result = self.request("/reset", {"seed": int(seed)})
        if not result.get("reset") or result.get("writes") != 0:
            raise RuntimeError("Policy memory reset was not acknowledged")

    def observe(self, raw_obs, index=-1):
        """Record EACH simulator frame, independently of action execution horizon."""
        if self.state_history_steps:
            state = observation(raw_obs, "", index)["observation/state"]
            self._states.append(state.copy())
        if self.image_history_frames:
            self._front.append(observation(raw_obs, "", index)["observation/image"].copy())

    def _with_history(self, obs):
        if self.state_history_steps:
            state = np.asarray(obs["observation/state"], np.float32)
            if not self._states:
                self._states.append(state.copy())
            if not np.array_equal(self._states[-1], state):
                raise ValueError("State history missing current frame: call observe on EVERY simulator frame")
            states = list(self._states)
            history = np.stack([states[max(0, len(states)-1-h*self.state_history_stride)]
                                for h in range(self.state_history_steps, 0, -1)])
            obs = dict(obs, **{"observation/state_history": history})
        if self.image_history_frames:
            front = np.asarray(obs["observation/image"])
            frames = list(self._front)
            if frames and not np.array_equal(frames[-1], front):
                raise ValueError("Image history missing current frame: call observe on EVERY simulator frame")
            if not frames:
                # first query of an episode arrives before its observe(): use the frame without storing it (observe
                # stores it right after), so the buffer never holds frame 0 twice
                frames = [front]
            extra, valid = {}, []
            for i in range(self.image_history_frames):
                back = (self.image_history_frames - i) * self.image_history_stride  # oldest first: t-16, ..., t-4
                pos = len(frames) - 1 - back
                valid.append(pos >= 0)
                extra["observation/history_{}".format(i)] = frames[max(0, pos)]  # LeRobot's clamp to the first frame
            if getattr(self, "mask_history", False):  # absent on clients built without __init__ (tests)
                valid = [False] * self.image_history_frames
            extra["observation/history_valid"] = np.asarray(valid, dtype=bool)
            obs = dict(obs, **extra)
        return obs

    def infer(self, obs):
        if self.metadata.get("diagnostic_oracle_subtask") or self.metadata.get("diagnostic_oracle_write"):
            raise ValueError("Oracle server must only be used by the explicit oracle diagnostic")
        return self.request("/infer", encode_observation(self._with_history(obs)))

    def infer_oracle_write(self, obs, subtask):
        """Stage-A style diagnostic: the model decodes freely, the TRUE sentence is written into its bank."""
        if not self.metadata.get("diagnostic_oracle_write"):
            raise ValueError("Normal policy server rejects oracle writes")
        return self.request("/infer", dict(encode_observation(self._with_history(obs)), oracle_write_subtask=subtask))

    def infer_oracle(self, obs, subtask):
        if not self.metadata.get("diagnostic_oracle_subtask"):
            raise ValueError("Normal policy server rejects oracle labels")
        return self.request("/infer", dict(encode_observation(self._with_history(obs)), diagnostic_subtask=subtask))
