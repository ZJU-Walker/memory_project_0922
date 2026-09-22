"""Re-render an eval video from its saved trace (09-15: RoboMME frames are 256 px wide, so the captions the eval renderer
draws below the frame ran off the right edge). Upscales the camera frame (default 3x -> 768 px) and wraps every caption
line; needs only the per-episode json, the dataset video and the sidecar, no model.

  .venv/bin/python cluster_robomme/rerender_video.py --json <dir>/ep07_oracle.json --out <dir>/ep07_oracle.mp4 \
      [--dataset robomme/data/lerobot/PickXtimes_official] [--sidecar ...] [--scale 3]
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import textwrap

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[2]
    p.add_argument("--json", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--dataset", type=pathlib.Path, default=root / "robomme/data/lerobot/PickXtimes_official")
    p.add_argument("--sidecar", type=pathlib.Path, default=root / "robomme/metadata/PickXtimes/subtasks_official.json")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()
    d = json.loads(args.json.read_text())
    ep = int(d["episode_index"]); sid = d["stable_id"]; mode = d["write_mode"]
    segs = json.loads(args.sidecar.read_text())["episodes"][sid]["segments"]
    length = max(s["end"] for s in segs) + 1
    frame_sentence = [""] * length
    for s in segs:
        for i in range(s["start"], s["end"] + 1):
            frame_sentence[i] = s["sentence"]
    records = d["records"]
    by_frame = {int(r["frame"]): r for r in records}
    stride = min(b - a for a, b in zip(sorted(by_frame), sorted(by_frame)[1:])) if len(by_frame) > 1 else 10
    video = args.dataset / "videos" / f"chunk-{ep // 1000:03d}" / "image" / f"episode_{ep:06d}.mp4"
    cap = cv2.VideoCapture(str(video))
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {video}")
    h0, w0 = img.shape[:2]
    width, height = w0 * args.scale, h0 * args.scale
    scale_font, line_h = 0.55, 24
    chars = max(20, int(width / (9.6 * scale_font / 0.55)))  # HERSHEY_SIMPLEX ~9.6 px per char at 0.55
    band = line_h * 9 + 16
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height + band}",
         "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22",
         str(args.out)], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    current = None
    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, img = cap.read()
        if not ok or frame_id >= length:
            break
        if frame_id in by_frame:
            current = by_frame[frame_id]
        canvas = np.zeros((height + band, width, 3), dtype=np.uint8)
        canvas[:height] = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        y = height + 22
        def put(text, color=(255, 255, 255), max_lines=2):
            nonlocal y
            for line in textwrap.wrap(text, chars)[:max_lines] or [""]:
                cv2.putText(canvas, line, (8, y), font, scale_font, color, 1, cv2.LINE_AA)
                y += line_h
        put(f"{sid}  frame {frame_id}  [{mode} writes]  prompt: {d['prompt']}", (200, 200, 200))
        put(f"GT phase : {frame_sentence[frame_id]}", (255, 255, 255), 1)
        if current is not None:
            ok_pred = current["pred"] == current["gt_target"]
            put(f"PRED @{current['frame']}: {current['pred']}   (conf {current.get('conf', 0):.2f})", (80, 220, 80) if ok_pred else (60, 60, 255), 2)
            put(f"target   : {current['gt_target']}" + ("   DECISION STEP" if current.get("decision") else ""), (180, 180, 180), 1)
            bank = current.get("bank") or []
            put(f"bank[{len(bank)}]: " + (" | ".join(bank[-3:]) if bank else "(empty)") + ("   <- WRITE" if current.get("written") else ""), (255, 200, 80), 2)
        if current is not None and current.get("decision") and frame_id - int(current["frame"]) < stride:
            cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (0, 200, 255), 3)
        proc.stdin.write(canvas.tobytes())
        frame_id += 1
    proc.stdin.close(); proc.wait()
    print(f"wrote {args.out} ({frame_id} frames, {width}x{height + band})")


if __name__ == "__main__":
    main()
