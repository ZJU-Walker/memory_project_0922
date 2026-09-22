#!/usr/bin/env python
"""Generate extra RoboMME expert demonstrations with the benchmark's own recorder.

This is the benchmark's dataset recipe (robomme_benchmark/tests/_shared/dataset_generation.py, `_run_one_episode`) driven
over many episodes and GPUs: for every episode `gym.make(env, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
render_mode="rgb_array", reward_mode="dense", seed=<base + attempt>, difficulty=<easy|medium|hard>)` is wrapped in
`RobommeRecordWrapper`, the fail-aware motion planner solves the env's `task_list`, and the recorder writes
`hdf5_files/<env>_ep<i>_seed<seed>.h5` when the episode succeeds (same layout as the released `record_dataset_<env>.h5`:
`episode_<i>/timestep_<k>/{obs,action,info}` + `episode_<i>/setup`). A failed attempt retries with seed + 1 (up to 30 times),
exactly as the benchmark helper does. The difficulty pattern (easy, easy, medium, hard repeating -> 50/25/25) and the
fail-recovery episodes (the helper switches recovery on for episodes 0-5: "z" for 0-2, "xy" for 3-5) mirror the released
train split, so the new episodes are distributed like the released 100. Episode numbers are LOCAL (0..N-1) so the helper's
recovery rule applies unchanged; the converter (prepare_robomme_h5_to_lerobot.py) renumbers them after the released set.

Driver (spawns one worker process per slot, pinned to a GPU through CUDA_VISIBLE_DEVICES, resumable):
  generate_robomme_demos.py --env BinFill --out /scr/kewalk/robomme_0920/gen/BinFill --episodes 100 --seed-base 24000 \
      --workers 8 --gpus 0,1
Outputs under --out: hdf5_files/*.h5 (one per successful episode; the recorder also writes its overlay mp4 per episode), records/episode_<i>.json, logs/worker_<k>.log and, when
every episode is done, record_dataset_<env>_metadata.json (the benchmark's metadata format: task, episode, seed, difficulty).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

BENCH = pathlib.Path(os.environ.get("ROBOMME_BENCHMARK_ROOT", "/iris/u/kewalk/robomme_benchmark"))
DIFFICULTY_PATTERN = ("easy", "easy", "medium", "hard")  # released train metadata: "eemh" repeated (50/25/25)
SEED_SPACING = 100  # released train seeds: 4000 + 100 * episode (+ attempt)


def difficulty_for(episode: int) -> str:
    return DIFFICULTY_PATTERN[episode % len(DIFFICULTY_PATTERN)]


def count_frames(h5_path: pathlib.Path) -> int:
    import h5py

    with h5py.File(h5_path, "r") as f:
        groups = [k for k in f if k.startswith("episode_")]
        if len(groups) != 1:
            raise RuntimeError(f"{h5_path}: expected one episode group, found {groups}")
        return sum(1 for k in f[groups[0]] if k.startswith("timestep_"))


def run_worker(args) -> None:
    sys.path.insert(0, str(BENCH))  # tests._shared.* and, through it, robomme.* (src on the path)
    sys.path.insert(0, str(BENCH / "src"))
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from tests._shared import dataset_generation as gen  # the benchmark's own recipe

    out = pathlib.Path(args.out)
    records = out / "records"
    records.mkdir(parents=True, exist_ok=True)
    hdf5_dir = out / "hdf5_files"
    for episode in [int(e) for e in args.worker_episodes.split(",") if e]:
        record_path = records / f"episode_{episode}.json"
        if record_path.exists():
            print(f"[ep {episode}] already done", flush=True)
            continue
        case = gen.DatasetCase(env_id=args.env, episode=episode, base_seed=args.seed_base + SEED_SPACING * episode,
                               difficulty=difficulty_for(episode), save_video=True, mode_tag="extra")
        # save_video=True is REQUIRED: RecordWrapper buffers HDF5 frames only inside its video branch
        # (`_video_should_record` = save_video and task != "NO RECORD"); with False every episode "succeeds" with 0 frames.
        t0 = time.monotonic()
        done = None
        for attempt in range(gen.MAX_SEED_ATTEMPTS):
            seed = case.base_seed + attempt
            h5_path = hdf5_dir / f"{args.env}_ep{episode}_seed{seed}.h5"
            ta = time.monotonic()
            try:
                ok = gen._run_one_episode(case=case, seed=seed, output_dir=out)  # noqa: SLF001
            except Exception as exc:  # the helper swallows these too and moves to the next seed
                print(f"[ep {episode}] seed {seed}: exception {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                ok = False
            if ok and h5_path.exists():
                try:
                    frames = count_frames(h5_path)
                except Exception as exc:
                    print(f"[ep {episode}] seed {seed}: unreadable H5 ({exc}); retrying", flush=True)
                    h5_path.unlink(missing_ok=True)
                    continue
                done = {"task": args.env, "episode": episode, "seed": seed, "difficulty": case.difficulty, "frames": frames,
                        "attempts": attempt + 1, "elapsed_s": round(time.monotonic() - t0, 1),
                        "h5": str(h5_path), "fail_recovery": bool(episode <= 5)}
                print(f"[ep {episode}] seed {seed}: SUCCESS {frames} frames in {time.monotonic() - ta:.0f}s "
                      f"(attempt {attempt + 1})", flush=True)
                break
            # the recorder opens its file at construction, so a failed attempt leaves an empty shell behind
            if h5_path.exists():
                h5_path.unlink()
            print(f"[ep {episode}] seed {seed}: failed after {time.monotonic() - ta:.0f}s", flush=True)
        if done is None:
            done = {"task": args.env, "episode": episode, "seed": None, "difficulty": case.difficulty, "frames": 0,
                    "attempts": gen.MAX_SEED_ATTEMPTS, "elapsed_s": round(time.monotonic() - t0, 1), "h5": None,
                    "fail_recovery": bool(episode <= 5), "error": f"no success in {gen.MAX_SEED_ATTEMPTS} seeds"}
            print(f"[ep {episode}] GAVE UP after {gen.MAX_SEED_ATTEMPTS} seeds", flush=True)
        tmp = record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(done, indent=2) + "\n")
        tmp.replace(record_path)


def run_driver(args) -> int:
    out = pathlib.Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(parents=True, exist_ok=True)
    episodes = [e for e in range(args.episodes) if not (out / "records" / f"episode_{e}.json").exists()]
    gpus = [g for g in args.gpus.split(",") if g != ""]
    workers = max(1, min(args.workers, len(episodes))) if episodes else 0
    print(f"{args.env}: {len(episodes)} episodes to generate with {workers} workers on GPUs {gpus} -> {out}", flush=True)
    procs = []
    for k in range(workers):
        mine = episodes[k::workers]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        log = open(out / "logs" / f"worker_{k}.log", "a")
        cmd = [sys.executable, "-u", __file__, "--env", args.env, "--out", str(out), "--seed-base", str(args.seed_base),
               "--worker-episodes", ",".join(map(str, mine))]
        procs.append((k, mine, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"  worker {k}: GPU {env['CUDA_VISIBLE_DEVICES']}, episodes {mine[:4]}{'...' if len(mine) > 4 else ''} "
              f"({len(mine)})", flush=True)
    t0 = time.monotonic()
    last = -1
    while any(p.poll() is None for _, _, p, _ in procs):
        time.sleep(30)
        n_done = sum(1 for e in range(args.episodes) if (out / "records" / f"episode_{e}.json").exists())
        if n_done != last:
            print(f"  progress {n_done}/{args.episodes} after {time.monotonic() - t0:.0f}s", flush=True)
            last = n_done
    for k, _, p, log in procs:
        log.close()
        if p.returncode != 0:
            print(f"  worker {k} exited with {p.returncode}", flush=True)
    recs = []
    for e in range(args.episodes):
        path = out / "records" / f"episode_{e}.json"
        if path.exists():
            recs.append(json.loads(path.read_text()))
    good = [r for r in recs if r.get("h5")]
    bad = [r for r in recs if not r.get("h5")]
    frames = [r["frames"] for r in good]
    summary = {"env": args.env, "requested": args.episodes, "succeeded": len(good), "failed": len(bad),
               "frames_total": sum(frames), "frames_min": min(frames) if frames else 0,
               "frames_max": max(frames) if frames else 0, "seed_base": args.seed_base,
               "mean_attempts": round(sum(r["attempts"] for r in good) / len(good), 2) if good else None,
               "elapsed_s": round(time.monotonic() - t0), "failed_episodes": [r["episode"] for r in bad]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if len(good) == args.episodes:
        meta = {"env_id": args.env, "record_count": len(good),
                "records": [{"task": r["task"], "episode": r["episode"], "seed": r["seed"], "difficulty": r["difficulty"]}
                            for r in good]}
        (out / f"record_dataset_{args.env}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if len(good) == args.episodes else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="BinFill")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=24000, help="episode i draws seeds seed_base + 100 i + attempt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--worker-episodes", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_episodes is not None:
        run_worker(args)
        return 0
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
