"""Convert released Counting HDF5 to lossless LeRobot videos and v6 label sidecars.

No relabeling, generated captions, extra simulator rollouts, or evaluation episodes.
Writes one episode at a time, with a completion marker only after all 100 episodes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
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

TASKS = ("BinFill", "PickXtimes", "SwingXtimes", "StopCube")
CAMERAS = {"image": "front_rgb", "left_wrist_image": "wrist_rgb"}


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
    return [{"start": s, "end": e - 1, "sentence": labels[s]}
            for s, e in zip(starts, starts[1:] + [len(labels)])]


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0), "max": x.max(0), "mean": x.mean(0),
            "std": x.std(0), "count": np.asarray([len(x)])}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--h5-root", type=Path, required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=100, help="Only 100-episode outputs register for training")
    args = p.parse_args()
    root, task = args.project_root, args.task
    meta_dir = root / "robomme/metadata" / task
    final_root = root / "robomme/data/lerobot" / task
    if (meta_dir / "prepared.json").exists():
        print(f"{task}: already prepared", flush=True)
        return
    source = args.h5_root / f"record_dataset_{task}.h5"
    local_h5 = args.scratch_root / "raw" / source.name
    local_h5.parent.mkdir(parents=True, exist_ok=True)
    if not local_h5.exists() or local_h5.stat().st_size != source.stat().st_size:
        print(f"{task}: staging {source.stat().st_size / 1e9:.1f} GB H5 to {local_h5}", flush=True)
        tmp = local_h5.with_suffix(".copying")
        shutil.copyfile(source, tmp)
        tmp.replace(local_h5)
    out = args.scratch_root / "lerobot" / task
    if out.exists():
        raise RuntimeError(f"Incomplete conversion exists: {out}; preserve/rename it before retrying")
    features = {key: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
                for key in CAMERAS}
    features.update({key: {"dtype": "float32", "shape": (8,), "names": [f"joint_{i}" for i in range(7)] + ["gripper"]}
                     for key in ("state", "actions")})
    meta = LeRobotDatasetMetadata.create(f"robomme/{task}", fps=30, root=out, robot_type="panda", features=features)
    hf_features = get_hf_features_from_features(meta.features)
    records, side_episodes, prompts, all_sentences = [], {}, {}, set()
    norm_rows = {"state": [], "actions": []}
    start_time = time.monotonic()
    with h5py.File(local_h5, "r") as h5:
        names = sorted(h5, key=lambda s: int(s.split("_")[-1]))
        assert len(names) == 100 and names == [f"episode_{i}" for i in range(100)]
        for ei, name in enumerate(names[:args.max_episodes]):
            ep = h5[name]
            step_names = sorted((s for s in ep if s.startswith("timestep_")), key=lambda s: int(s.split("_")[-1]))
            n = len(step_names)
            assert step_names == [f"timestep_{i}" for i in range(n)] and n > 0
            goals = [text(s) for s in np.asarray(ep["setup/task_goal"][()]).reshape(-1)]
            assert goals and all(g.strip() for g in goals)
            prompts[str(ei)] = goals[0]
            label_fields = {s: [] for s in ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online", "grounded_subgoal_online")}
            states, actions = [], []
            videos = {key: Video(out / meta.get_video_file_path(ei, key), 30) for key in CAMERAS}
            first_images = {}
            try:
                for ti, step_name in enumerate(step_names):
                    step = ep[step_name]
                    if bool(step["info/is_video_demo"][()]):
                        raise ValueError(f"Unexpected demonstration-only frame {task}/{name}/{step_name}")
                    state = np.concatenate((step["obs/joint_state"][()], step["obs/gripper_state"][()][:1])).astype(np.float32)
                    action = np.asarray(step["action/joint_action"][()], np.float32)
                    assert state.shape == action.shape == (8,)
                    assert np.isfinite(state).all() and np.isfinite(action).all()
                    assert action[-1] in (-1, 1)
                    states.append(state); actions.append(action)
                    for field in label_fields:
                        label = text(step[f"info/{field}"][()])
                        assert label.strip(), (task, name, ti, field)
                        label_fields[field].append(label)
                    for key, h5key in CAMERAS.items():
                        rgb = step[f"obs/{h5key}"][()]
                        videos[key].add(rgb)
                        if ti == 0:
                            first_images[key] = rgb
            finally:
                for video in videos.values():
                    video.close()
            for key in CAMERAS:
                with av.open(str(out / meta.get_video_file_path(ei, key))) as video:
                    stream = video.streams.video[0]
                    # LeRobot's pixel-channel helper omits GBR planar RGB (gbrp).
                    # Record the probed RGB metadata explicitly; retain lossless pixels.
                    meta.info["features"][key]["info"] = {
                        "video.height": stream.height, "video.width": stream.width,
                        "video.codec": stream.codec.canonical_name, "video.pix_fmt": stream.pix_fmt,
                        "video.is_depth_map": False, "video.fps": int(stream.base_rate),
                        "video.channels": 3, "has_audio": False,
                    }
                    decoded = next(video.decode(video=0)).to_ndarray(format="rgb24")
                assert np.array_equal(decoded, first_images[key]), "Lossless RGB video round-trip failed"
            labels = label_fields["simple_subgoal"]
            all_sentences.update(labels)
            for label in sorted(set(labels)):
                if meta.get_task_index(label) is None:
                    meta.add_task(label)
            arrays = {"state": np.stack(states), "actions": np.stack(actions),
                      "timestamp": np.arange(n, dtype=np.float32) / 30,
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
            stable_id = f"{task}/{name}"
            records.append({"episode_index": ei, "stable_id": stable_id, "split": "train", "include": True,
                            "class": text(ep["setup/difficulty"][()]), "expected_num_frames": n,
                            "seed": int(ep["setup/seed"][()]), "task_goals": goals})
            side_episodes[stable_id] = {"segments": segments(labels), "official_fields": {
                k: segments(v) for k, v in label_fields.items() if k != "simple_subgoal"}}
            print(f"{task}: episode {ei + 1}/{args.max_episodes}, {n} frames, total={meta.total_frames}, elapsed={time.monotonic()-start_time:.0f}s", flush=True)
    write_json(out / "meta/episode_prompts.json", prompts)
    manifest = {"schema_version": "openpi.v5.generic-manifest.v1", "split_seed": 0, "episodes": records,
                "source": str(source), "source_bytes": source.stat().st_size,
                "label_field": "info/simple_subgoal", "prompt_selection": "setup/task_goal[0]"}
    manifest_sha = write_json(meta_dir / "manifest.json", manifest)
    sidecar = {"schema_version": "openpi.v5.subtask-labels.v1", "source_manifest_sha256": manifest_sha,
               "episodes": side_episodes, "label_policy": "verbatim released simple_subgoal; original boundaries"}
    sidecar["content_sha256"] = hashlib.sha256((json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False)+"\n").encode()).hexdigest()
    subtasks_sha = write_json(meta_dir / "subtasks.json", sidecar)
    norm = {}
    for key, rows in norm_rows.items():
        values = np.concatenate(rows).astype(np.float64)
        norm[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                     "q01": np.quantile(values, .01, axis=0).tolist(), "q99": np.quantile(values, .99, axis=0).tolist()}
    norm_sha = write_json(root / "robomme/assets" / task / "norm_stats.json", {"norm_stats": norm})
    tokenizer_path = root / "v35/cache/openpi/big_vision/paligemma_tokenizer.model"
    tok = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    sentences = sorted(all_sentences)
    tokens = [tok.encode(s.lower().strip()) + tok.encode("\n") for s in sentences]
    assert max(map(len, tokens)) <= 48, "Official label exceeds existing v6 sentence buffer"
    if args.max_episodes != 100:
        print("Diagnostic subset complete; not registered as a training dataset", flush=True)
        return
    print(f"{task}: mirroring converted dataset to {final_root}", flush=True)
    shutil.copytree(out, final_root)
    spec = {"task": task, "episodes": len(records), "frames": meta.total_frames,
            "manifest_sha256": manifest_sha, "subtasks_sha256": subtasks_sha, "norm_stats_sha256": norm_sha,
            "sentences": sentences, "reference_tokens": tokens, "max_sentence_tokens": max(map(len, tokens)),
            "source": str(source), "source_bytes": source.stat().st_size, "local_dataset": str(out),
            "state": "joint_state[0:7] + gripper_state[0:1] (finger width in meters)",
            "actions": "absolute joint_action[0:8], gripper -1 close/+1 open", "video": "lossless RGB H264, 30fps indexing"}
    write_json(meta_dir / "prepared.json", spec)
    print(f"{task}: PREPARED {len(records)} episodes, {meta.total_frames} frames, {len(sentences)} sentences", flush=True)


if __name__ == "__main__":
    main()
