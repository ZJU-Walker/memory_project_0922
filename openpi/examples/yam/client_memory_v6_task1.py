"""Closed-loop robot client for the v6 task1 ("find the <object>") sentence-bank memory model on the bimanual YAM.

Derived from client_memory_v5.py (same YAM/RealSense plumbing, same RTC chunk broker, same
`scripts/serve_yam_memory.py` websocket contract). What changed for the v6 task1 checkpoints
(pi05_yam_mem_v6_task1B2 / B3; cluster_v6/README.md):

* memory clock: the task1 data ticks every 5 frames (server metadata memory_stride_frames=5), not 15
  -> steps_between_inference defaults to 5 and is validated against the server;
* prompts: "find the banana" | "find the box" | "find the spoon" | "find the tape" (--prompt);
* the bank holds the model's own notes ("spoon in bin 2"), then the closing note and "open bin k";
  the overlay shows the decoded sentence, its confidence and the bank (--show-memory);
* the server empties its bank ONLY on a {"reset_memory": true} ping: this client sends one before the
  ramp and one after it, so every run starts with a blank bank (press 'r' for a new episode).

Latency: the B2 server answers in ~260-300 ms. At 30 Hz that is ~9 control steps, above the RTC
maximum of 6, so the broker would block; --hz 20 (default) keeps one inference inside the trained
delay range. Use --hz 30 only if the server measures <= 200 ms.

Keys in the window:  r = reset the server-side memory AND fetch a fresh chunk
                     s = start recording the live view (top camera + overlay) to a new mp4
                     n = stop recording (that segment is finalised on disk)
                     q = stop (arms left in place).

Server (GPU box, H200 of job 17329416):
    JOB=17329416 GRES=1 GPU=0 NO_PLACEHOLDER=1 bash cluster_v6/serve_v6_job.sh <ckpt dir> pi05_yam_mem_v6_task1B2 8000

Client (robot computer; needs `gello_software`, `openpi_client` and opencv importable):
    python client_memory_v6_task1.py --host 10.79.12.149 --port 8000 --prompt "find the spoon"

Smoke-test the obs/action/subtask/memory contract without hardware:
    python client_memory_v6_task1.py --host 10.79.12.149 --port 8000 --dry-run
"""

import dataclasses
import datetime
import logging
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPTS = ("find the banana", "find the box", "find the spoon", "find the tape")

# Per-arm DOF: 6 arm joints + 1 gripper. Bimanual state/action is concat(left, right) -> 14.
BIMANUAL_DOF = 14


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.149"
    port: int = 8000
    ping_timeout: float = 600.0
    """Websocket keepalive timeout (s). Generous: a server started without --warmup compiles on
    the first request for minutes."""

    # --- Inference / control ---
    action_horizon: int = 50
    """Full model action horizon. Must match the server checkpoint."""
    steps_between_inference: int = 5
    """v6 task1 memory clock: replan (= one memory tick on the server) every 5 controls; validated against
    the server's memory_stride_frames. A larger value (e.g. 15) executes more of each chunk before the next
    inference: fewer server calls, so a slow tail-phase inference no longer stalls the loop, at the price of a
    coarser memory clock (a placement must stay visible for at least one tick to be written; the tail sentence
    can be up to one tick late). Needs --allow-tick-mismatch."""
    allow_tick_mismatch: bool = False
    """Accept steps_between_inference != the server's training memory tick (memory_stride_frames) with a warning."""
    initial_delay_steps: int = 6
    """Conservative initial latency estimate (6 steps = 200 ms at 30 Hz)."""
    max_async_delay_steps: int = 6
    """Never execute more unconfirmed steps than the largest RTC delay seen in training."""
    delay_tolerance_steps: int = 0
    """Block at the predicted delay rather than leaving the RTC-trained prefix range."""
    delay_buffer_size: int = 8
    """Number of measured inference delays retained by the conservative estimator."""
    max_steps: int = 12000
    hz: float = 20.0
    """Control rate. 20 keeps a ~300 ms inference inside the RTC delay range (6 steps); the training data
    ran at 30, so motions play ~1.5x slower at 20."""
    prompt: str = "find the spoon"
    """One of the task1 training prompts (see PROMPTS)."""
    max_joint_delta: float = 1.0
    """Per-step safety clamp: cap |target - current| across all joints to this many radians."""

    # --- Display ---
    show_memory: bool = False
    """Also put the memory readout (sees / bank) on the overlay bar; it is logged either way."""
    show: bool = True
    """Show the top camera + decoded sentence + bank contents in an OpenCV window."""

    # --- Hardware (defaults from gello configs/yam_left.yaml) ---
    can_left: str = "can_left"
    can_right: str = "can_right"
    top_camera_serial: str = "409122273280"
    left_camera_serial: str = "409122271088"
    right_camera_serial: str = "409122271086"

    # --- Recording (keyboard-driven: 's' starts a segment, 'n' stops it) ---
    record_on_start: bool = False
    """Start a recording at launch instead of waiting for the first 's'."""
    record_dir: str = "eval/memory_task_1_eval"
    """Directory for the recorded segments (one mp4 per s..n); created if missing."""
    record_path: str = ""
    """Explicit path for the first segment; later segments are timestamped in --record-dir."""

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: validate the RTC replan and obs/action/subtask/memory contract."""
    dry_run_steps: int = 40
    """Control steps in a dry run. Keep above 21 to exercise an asynchronous RTC replan."""


class _H264Writer:
    """Encode RGB frames to an H.264 mp4 via the system ffmpeg (same as client_memory.py)."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH -- needed to encode the recording")
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


def memory_readout(result: dict) -> tuple[str, str]:
    """Two overlay lines: the sentence decoded now (with confidence, `*` = committed this tick),
    and what the bank holds (the committed sentences, oldest first) with the commit count."""
    confidence = result.get("subtask_confidence")
    memory = result.get("memory") or {}
    mark = "*" if memory.get("committed") else ""
    conf = f"({float(confidence):.2f})" if confidence is not None else ""
    line_seen = f"sees: {result.get('subtask', '')!s}{conf}{mark}  (* = committed now)"
    bank = result.get("bank") or []
    shown = " | ".join(str(b) for b in bank) if bank else "-"
    line_held = f"bank[{len(bank)}]: {shown}  commits {result.get('writes', 0)}"
    return line_seen, line_held


class _Recorder:
    """Keyboard-driven segment recorder: one mp4 per 's' (start) .. 'n' (stop) pair.

    Frames are the same overlaid RGB images the window shows, written incrementally, so a
    segment survives Ctrl+C (the caller stops the recorder in its `finally`).
    """

    def __init__(self, record_dir: str, fps: float, first_path: str = "", prefix: str = "memory_v6_task1_closedloop"):
        self._dir = record_dir
        self._fps = fps
        self._first_path = first_path
        self._prefix = prefix
        self._writer: _H264Writer | None = None
        self._size: tuple[int, int] = (0, 0)  # (h, w) the encoder was opened with
        self._warned_resize = False
        self.path = ""
        self.frames = 0
        self.segments = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    def _next_path(self) -> str:
        if self._first_path and self.segments == 0:
            os.makedirs(os.path.dirname(os.path.abspath(self._first_path)) or ".", exist_ok=True)
            return self._first_path
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        os.makedirs(self._dir, exist_ok=True)
        # Segment index keeps two segments started within the same second distinct; the
        # existence check covers a second run of the client in that same second.
        base = os.path.join(self._dir, f"{self._prefix}_{stamp}_seg{self.segments + 1:02d}")
        path, dup = f"{base}.mp4", 1
        while os.path.exists(path):
            dup += 1
            path = f"{base}_{dup}.mp4"
        return path

    def start(self, frame_rgb: np.ndarray) -> None:
        if self._writer is not None:
            return
        self.path = self._next_path()
        self._size = (frame_rgb.shape[0], frame_rgb.shape[1])
        self._warned_resize = False
        self._writer = _H264Writer(self.path, frame_rgb.shape[1], frame_rgb.shape[0], self._fps)
        self.frames = 0
        logging.info("Recording started (H.264) -> %s @ %.1f Hz -- press 'n' to stop", self.path, self._fps)

    def _fit(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Force the encoder's frame size: the overlay bar grows/shrinks with the wrapped text,
        and raw video desyncs if a frame's byte count ever changes mid-segment."""
        h, w = self._size
        if frame_rgb.shape[:2] == (h, w):
            return frame_rgb
        if not self._warned_resize:
            self._warned_resize = True
            logging.warning(
                "overlay changed size mid-recording (%s -> %dx%d); padding/cropping to keep the mp4 valid",
                frame_rgb.shape[:2],
                h,
                w,
            )
        fitted = np.zeros((h, w, 3), dtype=frame_rgb.dtype)
        ih, iw = min(h, frame_rgb.shape[0]), min(w, frame_rgb.shape[1])
        fitted[:ih, :iw] = frame_rgb[:ih, :iw]
        return fitted

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(self._fit(frame_rgb))
            self.frames += 1

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        self.segments += 1
        logging.info("Saved recording: %s (%d frames)", self.path, self.frames)

    def label(self) -> str:
        return f"REC {self.frames / self._fps:5.1f}s" if self.active else "not recording"


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 12  # left/right/top/bottom margin inside the black bar


def _wrap(text: str, scale: float, thickness: int, max_width: int) -> list[str]:
    """Greedy word wrap to `max_width` pixels; long single words are split mid-word."""

    def width(s: str) -> int:
        return cv2.getTextSize(s, _FONT, scale, thickness)[0][0]

    out: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if width(candidate) <= max_width or not line:
            line = candidate
        else:
            out.append(line)
            line = word
        while width(line) > max_width and len(line) > 1:  # a word longer than the bar
            cut = len(line)
            while cut > 1 and width(line[:cut]) > max_width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
    if line:
        out.append(line)
    return out or [""]


# Rows the black bar always reserves. Fixed on purpose: the output frame size must never change
# mid-run, otherwise the raw-video pipe into ffmpeg desyncs (see _Recorder._fit).
_SUB_ROWS = 2
_SUB_SCALE, _SUB_THICK, _SUB_STEP = 0.8, 2, 30
_INFO_SCALE, _INFO_THICK, _INFO_STEP = 0.55, 1, 24


def _fit_rows(text: str, scale: float, thickness: int, max_width: int, rows: int) -> list[str]:
    """Wrap to `max_width` and clip to `rows` lines, marking a clipped last line with '...'."""
    wrapped = _wrap(text, scale, thickness, max_width)
    if len(wrapped) <= rows:
        return wrapped
    clipped = wrapped[:rows]
    clipped[-1] = clipped[-1][: max(0, len(clipped[-1]) - 3)] + "..."
    return clipped


def _overlay(frame_rgb: np.ndarray, subtask: str, lines: list[str]) -> np.ndarray:
    """Frame with a fixed-height black bar underneath: subtask (green) then `lines` (yellow).

    The bar sits below the camera image so nothing occludes the scene, text wraps to the frame
    width instead of running off the right edge, and the bar reserves a constant number of rows
    (`_SUB_ROWS` + len(lines)) so the composed frame is always exactly the same size.
    """
    frame = np.ascontiguousarray(frame_rgb)
    w = frame.shape[1]
    text_w = w - 2 * _PAD

    bar_h = 2 * _PAD + _SUB_STEP * _SUB_ROWS + _INFO_STEP * len(lines)
    bar = np.zeros((bar_h, w, 3), dtype=frame.dtype)

    y = _PAD
    for line in _fit_rows(f"subtask: {subtask}", _SUB_SCALE, _SUB_THICK, text_w, _SUB_ROWS):
        y += _SUB_STEP
        cv2.putText(bar, line, (_PAD, y - 6), _FONT, _SUB_SCALE, (90, 220, 120), _SUB_THICK, cv2.LINE_AA)
    y = _PAD + _SUB_STEP * _SUB_ROWS  # keep the info rows anchored, whatever the subtask wrapped to
    for line in lines:
        y += _INFO_STEP
        text = _fit_rows(line, _INFO_SCALE, _INFO_THICK, text_w, 1)[0]
        cv2.putText(bar, text, (_PAD, y - 5), _FONT, _INFO_SCALE, (235, 235, 60), _INFO_THICK, cv2.LINE_AA)
    return np.vstack([frame, bar])


class _Display:
    """Window thread showing the latest overlaid frame; captures 'r' (reset), 's'/'n' (start/stop
    recording the live view) and 'q' (quit)."""

    def __init__(self, window: str = "pi05 yam memory v6 task1 - closed loop"):
        self._window = window
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None  # RGB, already overlaid
        self.reset_requested = threading.Event()
        self.record_start_requested = threading.Event()
        self.record_stop_requested = threading.Event()
        self.quit_requested = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, img_rgb: np.ndarray) -> None:
        with self._lock:
            self._img = img_rgb

    def _loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                img = None if self._img is None else self._img.copy()
            if img is not None:
                cv2.imshow(self._window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.reset_requested.set()
            elif key == ord("s"):
                self.record_stop_requested.clear()
                self.record_start_requested.set()
            elif key == ord("n"):
                self.record_start_requested.clear()
                self.record_stop_requested.set()
            elif key == ord("q"):
                self.quit_requested.set()
        cv2.destroyAllWindows()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _obs_to_request(obs: dict, prompt: str) -> dict:
    """Map a `RobotEnv.get_obs()` dict to the websocket observation the server expects."""
    return {
        "observation/state": np.asarray(obs["joint_positions"], dtype=np.float32),
        "observation/image": image_tools.convert_to_uint8(obs["top_camera_rgb"]),
        "observation/left_wrist_image": image_tools.convert_to_uint8(obs["left_camera_rgb"]),
        "observation/right_wrist_image": image_tools.convert_to_uint8(obs["right_camera_rgb"]),
        "prompt": prompt,
    }


def _clamp_joint_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    """Scale the whole command so no single joint moves more than `max_delta` this step."""
    delta = target - current
    m = float(np.abs(delta).max())
    if m > max_delta:
        delta = delta / m * max_delta
    return current + delta


def validate_v6_metadata(metadata: dict, args: Args) -> None:
    """Fail before touching hardware when the client/server contracts differ."""
    if not metadata.get("memory_v5_sentence_bank"):
        raise ValueError(
            f"this client needs a v5/v6 sentence-bank server; server advertised config {metadata.get('config_name')!r} "
            f"with memory_v5_sentence_bank={metadata.get('memory_v5_sentence_bank')!r}"
        )
    if metadata.get("memory_architecture") != "v32_layer8_dual_query":
        raise ValueError(f"unexpected memory_architecture {metadata.get('memory_architecture')!r}")
    horizon = metadata.get("action_horizon")
    if horizon != args.action_horizon:
        raise ValueError(
            f"server action_horizon is {horizon!r}, but the client is configured for {args.action_horizon}"
        )
    if metadata.get("rtc_enabled") is not True:
        raise ValueError("the server checkpoint is not RTC-trained (rtc_enabled must be true)")
    if metadata.get("rtc_delay_semantics") != "inclusive_max":
        raise ValueError(f"unexpected RTC delay semantics {metadata.get('rtc_delay_semantics')!r}")
    trained_max_delay = metadata.get("rtc_max_delay")
    if not isinstance(trained_max_delay, int) or args.max_async_delay_steps > trained_max_delay:
        raise ValueError(
            f"client max_async_delay_steps={args.max_async_delay_steps} exceeds or cannot verify the "
            f"server's trained RTC maximum ({trained_max_delay!r})"
        )
    training_stride = metadata.get("memory_stride_frames")
    if training_stride != args.steps_between_inference:
        msg = (f"server memory_stride_frames is {training_stride!r}, but the client replans every "
               f"{args.steps_between_inference} steps (task1 v6 trained at 5)")
        if not args.allow_tick_mismatch:
            raise ValueError(msg + "; pass --allow-tick-mismatch to run with a coarser memory clock")
        logging.warning("%s -- running with a coarser memory clock (--allow-tick-mismatch)", msg)
    if args.prompt not in PROMPTS:
        raise ValueError(f"prompt {args.prompt!r} is not a training prompt; use one of {PROMPTS}")


def _run_dry(ws_client, policy, args: Args) -> None:
    """Validate the RTC replan and obs/action/subtask/memory contract without hardware."""
    if args.dry_run_steps <= args.steps_between_inference + args.max_async_delay_steps:
        raise ValueError(
            "dry_run_steps must exceed steps_between_inference + max_async_delay_steps "
            "so the test waits for at least one asynchronous RTC replan"
        )
    rng = np.random.default_rng(0)
    logging.info("Dry run: memory reset + %d random control observations...", args.dry_run_steps)
    logging.info("  reset: %s", ws_client.infer({"reset_memory": True}))
    for i in range(args.dry_run_steps):
        example = {
            "observation/state": rng.random(BIMANUAL_DOF).astype(np.float32),
            "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
        result = policy.infer(example)
        action = np.asarray(result["actions"])
        assert action.shape == (BIMANUAL_DOF,), f"expected (14,) per broker step, got {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action returned"
        assert isinstance(result.get("subtask"), str), f"missing subtask, got {result.get('subtask')!r}"
        for key in ("subtask_confidence", "bank", "memory", "writes"):
            assert key in result, f"v5 server response lacks {key!r}: {sorted(result)}"
        seen, held = memory_readout(result)
        logging.info("  step %d: subtask=%r | %s | %s", i, result["subtask"], seen, held)
    logging.info("Dry run OK -- RTC replan and obs/action/subtask/memory contract match.")


def main(args: Args) -> None:
    # --- Connect to the policy server ---
    try:
        ws_client = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port, ping_timeout=args.ping_timeout
        )
    except TypeError:
        # Older openpi_client on the robot computer (no ping_timeout): still works when the
        # server was started with --warmup (default), which keeps every request well under
        # the library's 20 s keepalive timeout.
        logging.warning("openpi_client without ping_timeout support; make sure the server was warmed up")
        ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    validate_v6_metadata(metadata, args)
    policy = action_chunk_broker.RealtimeActionChunkBroker(
        ws_client,
        action_horizon=args.action_horizon,
        steps_between_inference=args.steps_between_inference,
        initial_delay_steps=args.initial_delay_steps,
        delay_tolerance_steps=args.delay_tolerance_steps,
        max_async_delay_steps=args.max_async_delay_steps,
        delay_buffer_size=args.delay_buffer_size,
    )

    if args.dry_run:
        try:
            _run_dry(ws_client, policy, args)
        finally:
            policy.close()
        return

    # --- Build hardware (direct, in-process; same as client_memory.py) ---
    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.env import RobotEnv
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    left = YAMRobot(channel=args.can_left)
    right = YAMRobot(channel=args.can_right)
    robot = BimanualRobot(left, right)
    assert robot.num_dofs() == BIMANUAL_DOF, f"expected 14 DOF, got {robot.num_dofs()}"

    camera_dict = {
        "top_camera": RealSenseCamera(device_id=args.top_camera_serial),
        "left_camera": RealSenseCamera(device_id=args.left_camera_serial),
        "right_camera": RealSenseCamera(device_id=args.right_camera_serial),
    }
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=camera_dict)

    obs = env.get_obs()
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (avoid a large jump from rest) ---
    # The server memory survives websocket reconnects, so clear any previous client/episode
    # before even computing the ramp target. We reset once more after the ramp because that
    # first inference itself is a memory tick (and may commit if a fact is visible).
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset before computing the ramp target.")
    logging.info("Ramping to first policy target...")
    first_target = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))
    # Fresh chunk AND fresh banks for the episode (the ramp query already ticked once).
    policy.reset()
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset -- episode starts fresh (bank blank).")

    display = _Display() if args.show else None

    # --- Live-view recording: 's' starts a segment, 'n' stops it (written incrementally,
    # so Ctrl+C still saves whatever is open) ---
    recorder = _Recorder(args.record_dir, args.hz, first_path=args.record_path)
    # Segments open on the first overlaid frame of the loop, so the encoder is sized with the bar.
    pending_record = args.record_on_start
    if args.record_on_start:
        logging.info("Recording from launch (--record-on-start); 'n' stops it, 's' starts a new segment.")
    elif display is not None:
        logging.info("Press 's' in the window to start recording the live view, 'n' to stop.")
    else:
        logging.warning("--no-show: no window to capture keys, so 's'/'n' recording is unavailable")

    # --- Control loop (paced to `hz` by RobotEnv.step's internal Rate) ---
    logging.info(
        "Starting RTC control loop: %d steps @ %.1f Hz (replan/memory tick every %d steps, horizon %d), prompt %r",
        args.max_steps,
        args.hz,
        args.steps_between_inference,
        args.action_horizon,
        args.prompt,
    )
    obs = env.get_obs()
    subtask, seen, held = "", "", ""
    try:
        for step in range(args.max_steps):
            if display is not None and display.quit_requested.is_set():
                logging.info("Quit requested from the display window.")
                break
            if display is not None and display.reset_requested.is_set():
                display.reset_requested.clear()
                policy.reset()  # drop the cached chunk: it was computed with the old memory
                ws_client.infer({"reset_memory": True})
                logging.info("Memory reset (keyboard) -- new episode.")

            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            result = policy.infer(_obs_to_request(obs, args.prompt))
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))
            if "bank" in result:
                seen, held = memory_readout(result)
            status = f"step {step} | {recorder.label()} | r=reset s=rec n=stop q=quit"

            lines = [f"prompt: {args.prompt}", status]
            if args.show_memory:
                lines[1:1] = [seen, held]
            img = _overlay(frame, subtask, lines)
            if display is not None:
                display.update(img)
                if display.record_start_requested.is_set():
                    display.record_start_requested.clear()
                    pending_record = True
                if display.record_stop_requested.is_set():
                    display.record_stop_requested.clear()
                    recorder.stop()
            if pending_record:
                pending_record = False
                recorder.start(img)  # size the encoder from the overlaid frame (bar included)
            recorder.write(img)

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
            obs = env.step(action)
            if step % args.steps_between_inference == 0:
                logging.info("  step %d | subtask: %s | %s | %s", step, subtask, seen, held)
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        recorder.stop()
        if display is not None:
            display.close()
        policy.close()
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
