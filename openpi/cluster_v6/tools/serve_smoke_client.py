"""Workstation-side smoke for a running serve_yam_memory.py server: bare reset ping, then N synthetic observations
(random images, zero state, the prompt). Prints subtask / writes / latency per call. No robot needed.

  .venv/bin/python cluster_v6/tools/serve_smoke_client.py --host 10.79.12.149 --port 8000 --prompt "find the spoon" --state-dim 14
"""
import argparse, time
import numpy as np
from openpi_client import websocket_client_policy as wcp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="10.79.12.149"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="find the spoon"); ap.add_argument("--state-dim", type=int, default=14)
    ap.add_argument("--calls", type=int, default=4)
    a = ap.parse_args()
    pol = wcp.WebsocketClientPolicy(host=a.host, port=a.port)
    print("server metadata:", {k: v for k, v in (pol.get_server_metadata() or {}).items() if k in ("memory_architecture", "config", "action_horizon", "policy_step_hz")})
    t = time.monotonic(); r = pol.infer({"reset_memory": True}); print(f"reset ping -> {r} ({(time.monotonic()-t)*1e3:.0f} ms)")
    rng = np.random.default_rng(0)
    for i in range(a.calls):
        obs = {"observation/state": np.zeros(a.state_dim, np.float32),
               "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
               "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
               "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
               "prompt": a.prompt}
        t = time.monotonic(); r = pol.infer(obs); dt = (time.monotonic() - t) * 1e3
        acts = np.asarray(r["actions"]); print(f"call {i}: subtask={r.get('subtask')!r} conf={r.get('subtask_confidence', r.get('confidence'))} writes={r.get('writes')} actions={acts.shape} finite={bool(np.all(np.isfinite(acts)))} {dt:.0f} ms")


if __name__ == "__main__":
    main()
