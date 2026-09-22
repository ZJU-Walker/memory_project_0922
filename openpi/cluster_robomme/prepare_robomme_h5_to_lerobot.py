#!/usr/bin/env python
"""Convert released + extra RoboMME H5 demonstrations into ONE lossless LeRobot dataset with the official sentences.

Generalises prepare_legacy_h5_to_lerobot.py (one released 100-episode file, task == dataset name) to several sources and
any episode count, and folds in what build_pickxtimes_v2_labels.py --label-version official did afterwards for PickXtimes:
the per-frame task column holds the released `simple_subgoal` lowercased (verbatim otherwise, original boundaries), and
the sidecar/spec are written under the "official" tag, so `robomme_config._load_spec(root, "official", task=<name>)`
accepts them. No relabeling, no generated captions, no evaluation episodes.

Sources, in episode order: --released <record_dataset_<env>.h5> (episode_0..episode_99; staged to --scratch-root/raw first
when not already there) then --extra-dir <dir> (one H5 per episode from generate_robomme_demos.py, each holding a single
episode_<local> group, ordered by local episode number). Episodes are renumbered 0..N-1 in that order; stable ids keep the
source: "<env>/episode_<i>" for the released file, "<env>_extra/episode_<local>" for the generated ones.

Writes (under --project-root):
  robomme/data/lerobot/<name>/                    lossless RGB H264 videos (front + wrist), parquet with the task column
  robomme/metadata/<name>/manifest.json            openpi.v5.generic-manifest.v1 (every episode split "train")
  robomme/metadata/<name>/subtasks_official.json   openpi.v5.subtask-labels.v1 (segments lowercased, official_segments verbatim)
  robomme/metadata/<name>/prepared_official.json   spec: sentences, reference tokens, sha256 pins, lerobot_dataset, ...
  robomme/assets/<name>/norm_stats.json
usage: prepare_robomme_h5_to_lerobot.py --env BinFill --name BinFill200 --released <h5> --extra-dir <dir> \
         --expect-episodes 200 --project-root /iris/u/kewalk/memory_project_0920 --scratch-root /scr/kewalk/robomme_0920
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import av
import datasets
datasets.disable_progress_bar()
import h5py
import numpy as np
import sentencepiece
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_hf_features_from_features

CAMERAS = {"image": "front_rgb", "left_wrist_image": "wrist_rgb"}
LABEL_FIELDS = ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online", "grounded_subgoal_online")
FPS = 30  # indexing constant of every RoboMME conversion here (the simulator runs 20 control steps per second)


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return hashlib.sha256(body).hexdigest()


def segments(labels):
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    return [{"start": s, "end": e - 1, "sentence": labels[s]} for s, e in zip(starts, starts[1:] + [len(labels)])]


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.asarray([len(x)])}


class Video:
    def __init__(self, path, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264rgb", rate=fps)
        self.stream.width = self.stream.height = 256
        self.stream.pix_fmt = "rgb24"
        self.stream.options = {"crf": "0", "preset": "fast", "g": "10"}
        self.stream.codec_context.thread_count = 2

    def add(self, image):
        assert image.shape == (256, 256, 3) and image.dtype == np.uint8
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def stage(source: Path, scratch_root: Path) -> Path:
    local = scratch_root / "raw" / source.name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists() or local.stat().st_size != source.stat().st_size:
        print(f"staging {source.stat().st_size / 1e9:.1f} GB H5 to {local}", flush=True)
        tmp = local.with_suffix(".copying")
        shutil.copyfile(source, tmp)
        tmp.replace(local)
    return local


def list_sources(args):
    """[(h5 path, group name, stable id, source tag)] in dataset order."""
    items = []
    if args.released is not None:
        local = stage(args.released, args.scratch_root)
        with h5py.File(local, "r") as h5:
            names = sorted(h5, key=lambda s: int(s.split("_")[-1]))
        assert names == [f"episode_{i}" for i in range(len(names))], names[:3]
        items += [(local, name, f"{args.env}/{name}", "released") for name in names]
    if args.extra_dir is not None:
        files = sorted(args.extra_dir.glob(f"{args.env}_ep*_seed*.h5"),
                       key=lambda p: int(re.match(rf"{args.env}_ep(\d+)_seed", p.name).group(1)))
        locals_ = {}
        for f in files:
            local = int(re.match(rf"{args.env}_ep(\d+)_seed", f.name).group(1))
            assert local not in locals_, f"two files for local episode {local}: {locals_[local].name}, {f.name}"
            locals_[local] = f
        assert sorted(locals_) == list(range(len(locals_))), f"extra episodes are not contiguous: {sorted(locals_)[:5]}..."
        for local, f in sorted(locals_.items()):
            with h5py.File(f, "r") as h5:
                groups = [k for k in h5 if k.startswith("episode_")]
            assert groups == [f"episode_{local}"], (f.name, groups)
            items.append((f, groups[0], f"{args.env}_extra/episode_{local}", "extra"))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="RoboMME env id, e.g. BinFill")
    p.add_argument("--name", required=True, help="dataset / metadata / assets name, e.g. BinFill200")
    p.add_argument("--released", type=Path, default=None, help="released record_dataset_<env>.h5")
    p.add_argument("--extra-dir", type=Path, default=None, help="hdf5_files dir of generate_robomme_demos.py")
    p.add_argument("--expect-episodes", type=int, default=None)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    args = p.parse_args()
    root, name = args.project_root, args.name
    meta_dir = root / "robomme/metadata" / name
    final_root = root / "robomme/data/lerobot" / name
    if (meta_dir / "prepared_official.json").exists() or final_root.exists():
        raise RuntimeError(f"{name}: already prepared ({meta_dir} / {final_root}); remove or rename before rebuilding")
    items = list_sources(args)
    if args.expect_episodes is not None and len(items) != args.expect_episodes:
        raise RuntimeError(f"{name}: {len(items)} source episodes, expected {args.expect_episodes}")
    out = args.scratch_root / "lerobot" / name
    if out.exists():
        raise RuntimeError(f"Incomplete conversion exists: {out}; preserve/rename it before retrying")
    features = {key: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]} for key in CAMERAS}
    features.update({key: {"dtype": "float32", "shape": (8,), "names": [f"joint_{i}" for i in range(7)] + ["gripper"]}
                     for key in ("state", "actions")})
    meta = LeRobotDatasetMetadata.create(f"robomme/{name}", fps=FPS, root=out, robot_type="panda", features=features)
    hf_features = get_hf_features_from_features(meta.features)
    records, side_episodes, prompts, all_sentences = [], {}, {}, set()
    norm_rows = {"state": [], "actions": []}
    t0 = time.monotonic()
    for ei, (path, group, stable_id, source_tag) in enumerate(items):
        with h5py.File(path, "r") as h5:
            ep = h5[group]
            step_names = sorted((s for s in ep if s.startswith("timestep_")), key=lambda s: int(s.split("_")[-1]))
            n = len(step_names)
            assert step_names == [f"timestep_{i}" for i in range(n)] and n > 0, (stable_id, n)
            goals = [text(s) for s in np.asarray(ep["setup/task_goal"][()]).reshape(-1)]
            assert goals and all(g.strip() for g in goals), stable_id
            prompts[str(ei)] = goals[0]
            label_fields = {s: [] for s in LABEL_FIELDS}
            states, actions = [], []
            videos = {key: Video(out / meta.get_video_file_path(ei, key), FPS) for key in CAMERAS}
            first_images = {}
            try:
                for ti, step_name in enumerate(step_names):
                    step = ep[step_name]
                    if bool(step["info/is_video_demo"][()]):
                        raise ValueError(f"Unexpected demonstration-only frame {stable_id}/{step_name}")
                    state = np.concatenate((step["obs/joint_state"][()], step["obs/gripper_state"][()][:1])).astype(np.float32)
                    action = np.asarray(step["action/joint_action"][()], np.float32)
                    assert state.shape == action.shape == (8,), (stable_id, ti)
                    assert np.isfinite(state).all() and np.isfinite(action).all(), (stable_id, ti)
                    assert action[-1] in (-1, 1), (stable_id, ti, action[-1])
                    states.append(state)
                    actions.append(action)
                    for field in label_fields:
                        label = text(step[f"info/{field}"][()])
                        assert label.strip(), (stable_id, ti, field)
                        label_fields[field].append(label)
                    for key, h5key in CAMERAS.items():
                        rgb = step[f"obs/{h5key}"][()]
                        videos[key].add(rgb)
                        if ti == 0:
                            first_images[key] = rgb
            finally:
                for video in videos.values():
                    video.close()
            difficulty = text(ep["setup/difficulty"][()])
            seed = int(ep["setup/seed"][()])
        for key in CAMERAS:
            with av.open(str(out / meta.get_video_file_path(ei, key))) as video:
                stream = video.streams.video[0]
                meta.info["features"][key]["info"] = {
                    "video.height": stream.height, "video.width": stream.width,
                    "video.codec": stream.codec.canonical_name, "video.pix_fmt": stream.pix_fmt,
                    "video.is_depth_map": False, "video.fps": int(stream.base_rate), "video.channels": 3, "has_audio": False,
                }
                decoded = next(video.decode(video=0)).to_ndarray(format="rgb24")
            assert np.array_equal(decoded, first_images[key]), f"Lossless RGB video round-trip failed ({stable_id})"
        official = label_fields["simple_subgoal"]
        labels = [s.lower() for s in official]  # the official label policy: verbatim sentences, lowercased, same boundaries
        all_sentences.update(labels)
        for label in sorted(set(labels)):
            if meta.get_task_index(label) is None:
                meta.add_task(label)
        arrays = {"state": np.stack(states), "actions": np.stack(actions),
                  "timestamp": np.arange(n, dtype=np.float32) / FPS,
                  "frame_index": np.arange(n, dtype=np.int64),
                  "episode_index": np.full(n, ei, dtype=np.int64),
                  "index": np.arange(meta.total_frames, meta.total_frames + n, dtype=np.int64),
                  "task_index": np.asarray([meta.get_task_index(s) for s in labels], dtype=np.int64)}
        table = datasets.Dataset.from_dict(arrays, features=hf_features, split="train")
        parquet_path = out / meta.get_data_file_path(ei)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(parquet_path)
        meta.save_episode(ei, n, sorted(set(labels)), {k: stats(v) for k, v in arrays.items()})
        for key in norm_rows:
            norm_rows[key].append(arrays[key])
        records.append({"episode_index": ei, "stable_id": stable_id, "split": "train", "include": True, "class": difficulty,
                        "expected_num_frames": n, "seed": seed, "task_goals": goals, "source": source_tag,
                        "source_file": str(path), "source_group": group})
        side_episodes[stable_id] = {"class": difficulty, "num_frames": n, "segments": segments(labels),
                                    "official_segments": segments(official),
                                    "official_fields": {k: segments(v) for k, v in label_fields.items() if k != "simple_subgoal"}}
        print(f"{name}: episode {ei + 1}/{len(items)} ({source_tag} {group}), {n} frames, total={meta.total_frames}, "
              f"elapsed={time.monotonic() - t0:.0f}s", flush=True)
    write_json(out / "meta/episode_prompts.json", prompts)
    sources = sorted({str(path) if tag == "released" else str(path.parent) for path, _, _, tag in items})
    manifest = {"schema_version": "openpi.v5.generic-manifest.v1", "split_seed": 0, "episodes": records,
                "source": sources, "label_field": "info/simple_subgoal", "prompt_selection": "setup/task_goal[0]"}
    manifest_sha = write_json(meta_dir / "manifest.json", manifest)
    sentences = sorted(all_sentences)
    sidecar = {"schema_version": "openpi.v5.subtask-labels.v1", "dataset_version": f"{name}_official",
               "source_manifest": f"robomme/metadata/{name}/manifest.json", "source_manifest_sha256": manifest_sha,
               "num_episodes": len(side_episodes), "sentences": sentences, "episodes": side_episodes,
               "label_policy": "verbatim released simple_subgoal sentences and boundaries, lowercased; built by "
                               "cluster_robomme/prepare_robomme_h5_to_lerobot.py"}
    sidecar["content_sha256"] = hashlib.sha256(
        (json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()).hexdigest()
    subtasks_sha = write_json(meta_dir / "subtasks_official.json", sidecar)
    norm = {}
    for key, rows in norm_rows.items():
        values = np.concatenate(rows).astype(np.float64)
        norm[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                     "q01": np.quantile(values, .01, axis=0).tolist(), "q99": np.quantile(values, .99, axis=0).tolist()}
    norm_sha = write_json(root / "robomme/assets" / name / "norm_stats.json", {"norm_stats": norm})
    tok = sentencepiece.SentencePieceProcessor(model_file=str(root / "v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    tokens = [tok.encode(s.lower().strip()) + tok.encode("\n") for s in sentences]
    assert max(map(len, tokens)) <= 48, "Official label exceeds the 48-token sentence buffer"
    print(f"{name}: mirroring converted dataset to {final_root}", flush=True)
    shutil.copytree(out, final_root)
    spec = {"task": name, "env": args.env, "label_version": "official", "episodes": len(records), "frames": meta.total_frames,
            "manifest_sha256": manifest_sha, "subtasks_official_sha256": subtasks_sha, "norm_stats_sha256": norm_sha,
            "sentences": sentences, "reference_tokens": tokens, "max_sentence_tokens": max(map(len, tokens)),
            "max_segments_per_episode": max(len(e["segments"]) for e in side_episodes.values()),
            "lerobot_dataset": f"robomme/data/lerobot/{name}", "sources": sources, "local_dataset": str(out),
            "episodes_by_source": {tag: sum(1 for r in records if r["source"] == tag) for tag in ("released", "extra")},
            "state": "joint_state[0:7] + gripper_state[0:1] (finger width in meters)",
            "actions": "absolute joint_action[0:8], gripper -1 close/+1 open; observation row t supervises actions from t+1",
            "video": "lossless RGB H264, 30fps indexing", "videos_copied": True}
    write_json(meta_dir / "prepared_official.json", spec)
    print(f"{name}: PREPARED {len(records)} episodes, {meta.total_frames} frames, {len(sentences)} sentences, "
          f"max {spec['max_segments_per_episode']} segments/episode", flush=True)


if __name__ == "__main__":
    main()
