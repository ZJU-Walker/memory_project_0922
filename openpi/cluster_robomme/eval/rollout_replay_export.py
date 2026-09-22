"""Export a recorded rollout (predictions.jsonl + frames.jsonl) in the motion viewer's format.

Per tick: the note, the simulator's true subtask at that frame (color coding), the "Last:" prompt slot, the commit
flag, the bank, the action expert's chunk, the FAST chunk (when the server ran with --fast-decode) and, instead of a
demo's ground truth, the path the arm actually went on to execute over the next horizon frames. Joint chunks and the
executed path are converted to tool-center xyz with panda_fk.

    rollout_replay_export.py <rollout_dir> --out <dir> [--name rollout_v2B1000_ep03]
"""
import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np

import panda_fk
from demo_replay import pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rollout", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--horizon", type=int, default=50)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((args.rollout / "manifest.json").read_text())
    frames = [json.loads(l) for l in (args.rollout / "frames.jsonl").read_text().splitlines() if l.strip()]
    preds = [json.loads(l) for l in (args.rollout / "predictions.jsonl").read_text().splitlines() if l.strip()]
    # one row per simulator step (the last observation of each step carries the true subtask)
    by_step = {}
    for r in frames:
        if "gt_simple" in r or r["step"] not in by_step:
            by_step[r["step"]] = r
    steps = sorted(by_step)
    states = np.asarray([by_step[s]["robot_state"] for s in steps], dtype=np.float32)  # [n, 8]
    labels = [(by_step[s].get("gt_simple") or "").strip().lower() for s in steps]
    path_xyz = panda_fk.tcp_positions(states[:, :7])
    pol = m.get("policy") or {}
    name = args.name or f"rollout_{pol.get('training_config', 'policy')}_{pol.get('trained_update')}_ep{int(m['episode']):02d}"
    ticks = []
    for q in preds:
        if q.get("phase") != "control":
            continue
        s = int(q["step"])
        if s not in by_step:
            continue
        i = steps.index(s)
        ex = np.asarray(q["actions"], dtype=np.float32)
        fast = q.get("fast_actions")
        executed = states[i : i + args.horizon]
        note = (q.get("subtask") or "").strip()
        row = {
            "f": s, "note": note, "label": labels[i], "ok": note.lower() == labels[i],
            "prompt": q.get("prompt_prev_subtask"), "committed": bool((q.get("memory") or {}).get("committed", False)),
            "changed": bool((q.get("memory") or {}).get("changed", False)), "bank": list(q.get("bank") or []),
            "writes": int(q.get("writes", 0)), "conf": q.get("subtask_confidence"),
            "cur": np.round(path_xyz[i], 4).tolist(), "grip": round(float(states[i, 7]), 4),
            "ex": pack(ex, args.horizon), "gt": pack(executed, args.horizon),
            "fast": pack(np.asarray(fast, dtype=np.float32), args.horizon) if fast is not None else None,
            "fast_ok": q.get("fast_ok"), "lat": round(float(q.get("latency_ms", 0))),
        }
        gt_xyz = np.asarray(row["gt"]["xyz"])
        row["d_ex_gt"] = round(float(np.linalg.norm(np.asarray(row["ex"]["xyz"]) - gt_xyz, axis=-1).mean()), 4)
        if fast is not None:
            row["d_fast_gt"] = round(float(np.linalg.norm(np.asarray(row["fast"]["xyz"]) - gt_xyz, axis=-1).mean()), 4)
        ticks.append(row)
    video = args.out / f"{name}.mp4"
    src = args.rollout / "rollout.mp4"
    if src.exists() and not video.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=768:-2", "-c:v", "libx264", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)
    exact = float(np.mean([t["ok"] for t in ticks])) * 100 if ticks else 0.0
    out = {
        "meta": {"kind": "rollout", "episode": int(m["episode"]), "goal": m.get("goal", ""), "frames": len(steps), "fps": int(m.get("fps", 30)),
                 "tick": int(m.get("action_chunk") or 15), "horizon": args.horizon, "exec": int(m.get("execute_horizon") or m.get("action_chunk") or 0),
                 "control_hz": int(m.get("control_hz") or 20), "status": m.get("status"), "mode": m.get("mode"),
                 "policy": {"training_config": pol.get("training_config"), "trained_update": pol.get("trained_update")},
                 "video": video.name, "queries": len(ticks), "exact_pct": round(exact, 1),
                 "mean_d_ex_gt": round(float(np.mean([t["d_ex_gt"] for t in ticks])), 4) if ticks else None},
        "path": {"xyz": np.round(path_xyz, 4).tolist(), "g": np.round(states[:, 7], 4).tolist(), "label": labels},
        "ticks": ticks,
    }
    (args.out / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: {len(ticks)} ticks, status {m.get('status')}, notes exact {exact:.1f}%, fast {'yes' if any(t['fast'] for t in ticks) else 'no'}")


if __name__ == "__main__":
    main()
