"""Replay training demos through the policy server at the memory tick and export everything the motion viewer needs.

For every tick (every --chunk-size frames = 2 Hz at 30 fps) the recorded frames go to the server exactly as in the
expert-play test (own notes, memory reset at the episode start) and we keep: the decoded note, the training label of
that frame (the dataset's task string), the "Last:" prompt slot content, the commit flag and the bank, the action
expert's chunk, the FAST chunk (server started with --fast-decode) and the demo's own next actions. Joint chunks are
also converted to tool-center xyz with panda_fk. Nothing is executed.

    demo_replay.py --policy-url http://127.0.0.1:18990 --dataset <root>/robomme/data/lerobot/PickXtimes_official \
        --episodes 3,17,42 --out <dir>        (or --random 3 --seed 0 to draw episodes)
    demo_replay.py --policy-url ... --records <root>/robomme/expert_val --out <dir>   (held-out expert plays: the five
        val scenes recorded with the benchmark's planner; reference = the next recorded joint states, gripper command
        read off the finger opening; labels = the recording's official sentences)
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np

from common import HttpPolicyClient, jsonable, observation
import panda_fk


def read_video(path):
    import imageio.v3 as iio

    return np.asarray(iio.imread(str(path), plugin="pyav"))


def read_episode(root, ep):
    import pyarrow.parquet as pq

    chunk = ep // 1000
    table = pq.read_table(root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet").to_pandas()
    state = np.stack(table["state"].to_numpy()).astype(np.float32)
    actions = np.stack(table["actions"].to_numpy()).astype(np.float32)
    task_index = table["task_index"].to_numpy().astype(int)
    front = read_video(root / "videos" / f"chunk-{chunk:03d}" / "image" / f"episode_{ep:06d}.mp4")
    wrist = read_video(root / "videos" / f"chunk-{chunk:03d}" / "left_wrist_image" / f"episode_{ep:06d}.mp4")
    n = min(len(state), len(front), len(wrist))
    return state[:n], actions[:n], task_index[:n], front[:n], wrist[:n]


def read_expert(records, ep):
    """One held-out expert recording (record_expert_val.py): frames.npz + meta.json -> the demo-like arrays."""
    d = records / f"PickXtimes_val_ep{ep:03d}"
    z = np.load(d / "frames.npz")
    meta = json.loads((d / "meta.json").read_text())
    joint, grip = z["joint"].astype(np.float32), z["gripper"].astype(np.float32)
    state = np.concatenate([joint, grip[:, :1]], axis=1)
    # reference targets: the planner's next recorded states; gripper command +1 when the fingers are open (> 3 cm), else -1
    nxt = np.concatenate([state[1:], state[-1:]], axis=0)
    cmd = np.where(nxt[:, 7] > 0.03, 1.0, -1.0).astype(np.float32)
    actions = np.concatenate([nxt[:, :7], cmd[:, None]], axis=1)
    labels = [(l or "").strip().lower() for l in meta["labels"]]
    n = min(len(state), len(z["front"]), len(z["wrist"]), len(labels))
    return dict(kind="expert", ep=ep, goal=meta["goal"], state=state[:n], actions=actions[:n], labels=labels[:n],
                front=z["front"][:n], wrist=z["wrist"][:n], fps=int(meta.get("fps", 30)), video_src=d / "preview.mp4",
                extra={"expert_status": meta.get("final_status"), "split": meta.get("split")})


def chunk_of(arr, f, horizon):
    seg = arr[f : f + horizon]
    if len(seg) < horizon:
        seg = np.concatenate([seg, np.repeat(seg[-1:], horizon - len(seg), axis=0)], axis=0)
    return seg


def pack(actions, horizon):
    """Joint chunk [h, 8] -> {"xyz": [[x, y, z] * h], "g": [h], "q": [[8] * h]} (xyz = tool center in the base frame)."""
    a = np.asarray(actions, dtype=np.float32)[:horizon]
    if len(a) < horizon:
        a = np.concatenate([a, np.repeat(a[-1:], horizon - len(a), axis=0)], axis=0)
    xyz = panda_fk.tcp_positions(a[:, :7])
    return {"xyz": np.round(xyz, 4).tolist(), "g": np.round(a[:, 7], 3).tolist(), "q": np.round(a, 4).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-url", required=True)
    p.add_argument("--dataset", type=pathlib.Path, default=None, help="LeRobot training set (training demos)")
    p.add_argument("--records", type=pathlib.Path, default=None, help="expert_val recordings (held-out expert plays)")
    p.add_argument("--val-episodes", default="", help="comma list of val episode indices (default: every recording)")
    p.add_argument("--episodes", default="", help="comma list of episode indices")
    p.add_argument("--random", type=int, default=0, help="draw this many training episodes instead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sources = []
    if args.dataset is not None:
        info = json.loads((args.dataset / "meta" / "info.json").read_text())
        fps = int(info.get("fps", 30))
        tasks = {}
        for line in (args.dataset / "meta" / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                tasks[int(j["task_index"])] = str(j["task"])
        prompts = json.loads((args.dataset / "meta" / "episode_prompts.json").read_text())
        total = int(info["total_episodes"])
        if args.random:
            rng = np.random.default_rng(args.seed)
            episodes = sorted(int(e) for e in rng.choice(total, size=args.random, replace=False))
        else:
            episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
        for ep in episodes:
            state, actions, task_index, front, wrist = read_episode(args.dataset, ep)
            chunk = ep // 1000
            sources.append(dict(kind="demo", ep=ep, goal=prompts[str(ep)], state=state, actions=actions, labels=[tasks.get(int(t), "") for t in task_index],
                                front=front, wrist=wrist, fps=fps, extra={},
                                video_src=(args.dataset / "videos" / f"chunk-{chunk:03d}" / "image" / f"episode_{ep:06d}.mp4",
                                           args.dataset / "videos" / f"chunk-{chunk:03d}" / "left_wrist_image" / f"episode_{ep:06d}.mp4")))
    if args.records is not None:
        found = sorted(int(d.name.split("_ep")[-1]) for d in args.records.glob("PickXtimes_val_ep*") if (d / "frames.npz").exists())
        wanted = [int(e) for e in args.val_episodes.split(",") if e.strip()] or found
        for ep in wanted:
            sources.append(read_expert(args.records, ep))
    if not sources:
        raise SystemExit("nothing to replay: give --dataset with --episodes/--random and/or --records")
    print(f"sources {[(s['kind'], s['ep']) for s in sources]} | tick {args.chunk_size} frames | horizon {args.horizon}", flush=True)

    client = HttpPolicyClient(args.policy_url, args.task, args.chunk_size)
    meta = client.metadata
    print(f"server: {meta.get('training_config')} update {meta.get('trained_update')} fast_decode={meta.get('fast_decode')}", flush=True)
    summary = []
    for src in sources:
        ep, goal, state, actions, front, wrist, fps = src["ep"], src["goal"], src["state"], src["actions"], src["front"], src["wrist"], src["fps"]
        frame_labels = src["labels"]
        n = len(state)
        client.reset(args.seed)
        ticks = []
        t_ep = time.monotonic()
        for f in range(n):
            g = float(state[f, 7])
            ob = {"front_rgb_list": [front[f]], "wrist_rgb_list": [wrist[f]], "joint_state_list": [state[f, :7]], "gripper_state_list": [np.array([g, g], dtype=np.float32)]}
            client.observe(ob)
            if f % args.chunk_size:
                continue
            t0 = time.monotonic()
            pred = client.infer(observation(ob, goal))
            label = frame_labels[f]
            note = (pred.get("subtask") or "").strip()
            ex = np.asarray(pred["actions"], dtype=np.float32)
            fast = pred.get("fast_actions")
            gt = chunk_of(actions, f, args.horizon)
            row = {
                "f": int(f), "note": note, "label": label, "ok": note == label.strip().lower(),
                "prompt": pred.get("prompt_prev_subtask"), "committed": bool((pred.get("memory") or {}).get("committed", False)),
                "changed": bool((pred.get("memory") or {}).get("changed", False)), "bank": list(pred.get("bank") or []),
                "writes": int(pred.get("writes", 0)), "conf": pred.get("subtask_confidence"),
                "cur": np.round(panda_fk.tcp_positions(state[f, :7]), 4).tolist(), "grip": round(g, 4),
                "ex": pack(ex, args.horizon), "gt": pack(gt, args.horizon),
                "fast": pack(np.asarray(fast, dtype=np.float32), args.horizon) if fast is not None else None,
                "fast_ok": pred.get("fast_ok"), "lat": round((time.monotonic() - t0) * 1000),
            }
            # chunk disagreement in tool-center space (m, mean over the horizon) and in joint space (rad, rms)
            gt_xyz = np.asarray(row["gt"]["xyz"]); ex_xyz = np.asarray(row["ex"]["xyz"])
            row["d_ex_gt"] = round(float(np.linalg.norm(ex_xyz - gt_xyz, axis=-1).mean()), 4)
            row["q_ex_gt"] = round(float(np.sqrt(np.mean((ex[:, :7] - gt[:, :7]) ** 2))), 4)
            if fast is not None:
                fa = np.asarray(fast, dtype=np.float32)
                row["d_fast_gt"] = round(float(np.linalg.norm(np.asarray(row["fast"]["xyz"]) - gt_xyz, axis=-1).mean()), 4)
                row["q_fast_gt"] = round(float(np.sqrt(np.mean((fa[:, :7] - gt[:, :7]) ** 2))), 4)
            ticks.append(row)
            print(f"ep{ep} f={f:>4} note={note!r} label={label!r} ok={row['ok']} prompt={row['prompt']!r} commit={row['committed']} "
                  f"|ex-gt| {row['d_ex_gt']:.3f} m" + (f" |fast-gt| {row['d_fast_gt']:.3f} m" if fast is not None else "") + f" ({row['lat']} ms)", flush=True)
        path_xyz = panda_fk.tcp_positions(state[:, :7])
        exact = float(np.mean([t["ok"] for t in ticks])) * 100 if ticks else 0.0
        s = {"episode": ep, "goal": goal, "frames": n, "queries": len(ticks), "exact_pct": round(exact, 1),
             "mean_d_ex_gt": round(float(np.mean([t["d_ex_gt"] for t in ticks])), 4),
             "mean_d_fast_gt": round(float(np.mean([t["d_fast_gt"] for t in ticks if "d_fast_gt" in t])), 4) if any("d_fast_gt" in t for t in ticks) else None,
             "seconds": round(time.monotonic() - t_ep)}
        summary.append(s)
        name = f"{src['kind']}_ep{ep:03d}"
        video = args.out / f"{name}.mp4"
        if not args.no_video and not video.exists():
            vs = src["video_src"]
            if isinstance(vs, tuple):
                fv, wv = vs
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(fv), "-i", str(wv), "-filter_complex",
                                "[0:v]scale=384:-2[a];[1:v]scale=384:-2[b];[a][b]hstack", "-r", str(fps), "-c:v", "libx264", "-crf", "28",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)
            else:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vs), "-vf", "scale=768:-2", "-r", str(fps), "-c:v", "libx264", "-crf", "28",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)
        (args.out / f"{name}.json").write_text(json.dumps(jsonable({
            "meta": {"kind": src["kind"], "episode": ep, "goal": goal, "frames": n, "fps": fps, "tick": args.chunk_size, "horizon": args.horizon,
                     "policy": {k: meta.get(k) for k in ("training_config", "trained_update", "config_name", "fast_decode")}, "video": video.name, **s, **src["extra"]},
            "path": {"xyz": np.round(path_xyz, 4).tolist(), "g": np.round(state[:, 7], 4).tolist(), "label": frame_labels},
            "ticks": ticks})))
        print(f"== ep{ep}: {len(ticks)} ticks, notes exact {exact:.1f}%, mean |expert-gt| {s['mean_d_ex_gt']} m, mean |fast-gt| {s['mean_d_fast_gt']} m, {s['seconds']} s", flush=True)
    (args.out / "summary.json").write_text(json.dumps({"policy": meta, "episodes": summary}, indent=1))


if __name__ == "__main__":
    main()
