"""PickXtimes two-phase labels (v2) + the LeRobot dataset whose task column carries them.

User 2026-09-13 18:22: "same thing as the scoop in the boba, relabel the subtask in the robomme pick X times so each pick
has 2 subtasks". The official `simple_subgoal` already cuts every pick-and-place cycle into two segments -- `pick up the
<color> cube for the <k-th> time` (reach, grasp, lift) and `place the <color> cube onto the target` (carry, release,
retreat) -- but only the pick sentence carries the count. The v7 pointer read returns the LAST written sentence, so at
the next pick onset the bank held a count-free sentence and the count had to be recovered from two writes back. v2 keeps
the official boundaries (planner view) and gives the place phase the count of its pick:

    pick up the red cube for the second time  ->  place the red cube onto the target for the second time

Every transition is now "last written sentence -> same count, other phase" (pick k -> place k, on the loud grasp+lift)
or "last + 1" (place k -> pick k+1, on the loud release+retreat), exactly the boba v2 scoop/pour pattern. Sentences are
lowercased (the reference tokens are `sp.encode(sentence.lower().strip()) + sp.encode("\\n")`, and the base decodes the
task column verbatim).

Inputs: the legacy lossless LeRobot conversion (memory_v6_robomme, pixel-exact H264, verified) and its official-label
sidecar / manifest, already copied to robomme/metadata/PickXtimes. Outputs (all under robomme/):
  * metadata/PickXtimes/subtasks_v2.json  (openpi.v5.subtask-labels.v1 sidecar, official segments kept for inspection)
  * metadata/PickXtimes/prepared_v2.json  (pinned SHA256s, sentence list, PaliGemma reference tokens)
  * data/lerobot/PickXtimes_v2/           (videos copied, parquet task_index + meta rewritten to the v2 vocabulary)
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece

ORDINALS = ("first", "second", "third", "fourth", "fifth")
PICK = re.compile(r"^pick up the (\w+) cube for the (\w+) time$")
PLACE = re.compile(r"^place the (\w+) cube onto the target$")
OTHER = ("press the button to stop", "all tasks completed")
SIDECAR_SCHEMA = "openpi.v5.subtask-labels.v1"


def write_json(path: pathlib.Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return hashlib.sha256(body).hexdigest()


def relabel(segments: list[dict], stable_id: str) -> list[dict]:
    out, ordinal, color, expect_pick, k = [], None, None, True, 0
    for seg in segments:
        s = seg["sentence"].strip().lower()
        if m := PICK.match(s):
            if not expect_pick:
                raise ValueError(f"{stable_id}: two picks in a row at frame {seg['start']}")
            color, ordinal = m.group(1), m.group(2)
            if ordinal != ORDINALS[k]:
                raise ValueError(f"{stable_id}: pick ordinal {ordinal!r} at frame {seg['start']}, expected {ORDINALS[k]!r}")
            k += 1
            expect_pick = False
            new = s
        elif m := PLACE.match(s):
            if expect_pick or m.group(1) != color:
                raise ValueError(f"{stable_id}: place without a matching pick at frame {seg['start']}")
            expect_pick = True
            new = f"place the {color} cube onto the target for the {ordinal} time"
        elif s in OTHER:
            if not expect_pick:
                raise ValueError(f"{stable_id}: {s!r} inside an unfinished cycle at frame {seg['start']}")
            new = s
        else:
            raise ValueError(f"{stable_id}: unexpected official sentence {seg['sentence']!r}")
        out.append({"start": int(seg["start"]), "end": int(seg["end"]), "sentence": new})
    if [x["sentence"] for x in out[-2:]] != list(OTHER):
        raise ValueError(f"{stable_id}: episode does not end with press + completed: {[x['sentence'] for x in out[-2:]]}")
    return out


def relabel_official(segments: list[dict], stable_id: str) -> list[dict]:
    """09-15 (user 15:00 "i still want to use the official provided subtask"): the released simple_subgoal sentences
    verbatim except lowercased (the reference tokens are encoded from the lowercased sentence); `relabel` only
    validates the cycle structure."""
    relabel(segments, stable_id)
    return [{"start": int(s["start"]), "end": int(s["end"]), "sentence": s["sentence"].strip().lower()} for s in segments]


def relabel_shift1(segments: list[dict], stable_id: str) -> list[dict]:
    """COUNTERFACTUAL (09-15 18:50, eval only, never training): the official sentences with every pick ordinal moved
    one up (first -> second ... fourth -> fifth); episodes with five picks are left unchanged. Fed to the oracle-write
    pass through a dataset + sidecar of its own because the loader, not the eval flag, supplies the oracle tokens."""
    out = relabel_official(segments, stable_id)
    if any(PICK.match(s["sentence"]).group(2) == ORDINALS[-1] for s in out if PICK.match(s["sentence"])):
        return out
    for seg in out:
        if m := PICK.match(seg["sentence"]):
            seg["sentence"] = f"pick up the {m.group(1)} cube for the {ORDINALS[ORDINALS.index(m.group(2)) + 1]} time"
    return out


def relabel_tgt(segments: list[dict], stable_id: str) -> list[dict]:
    """Target-carry two-phase sentences (09-17, user: replicate the bean-scoop B9 recipe). Every pick and place
    sentence names its own count k AND the episode's target x, the number of picks the goal asks for (= the number
    of pick segments; checked against the prompt by the caller), so the press decision is "the previous sentence says
    x of x" instead of a comparison between the bank and the prompt (the beans fix of 2026-09-05, v7tgt labels):

        pick up the red cube, 2 of 3   ->   place the red cube onto the target, 2 of 3   ->   ... press ... completed
    """
    two_phase = relabel(segments, stable_id)  # validates the cycle structure, place carries its pick's ordinal
    x = sum(1 for seg in two_phase if PICK.match(seg["sentence"]))
    if not 1 <= x <= len(ORDINALS):
        raise ValueError(f"{stable_id}: {x} picks")
    out = []
    for seg in two_phase:
        s = seg["sentence"]
        if m := PICK.match(s):
            new = f"pick up the {m.group(1)} cube, {ORDINALS.index(m.group(2)) + 1} of {x}"
        elif m := re.match(r"^place the (\w+) cube onto the target for the (\w+) time$", s):
            new = f"place the {m.group(1)} cube onto the target, {ORDINALS.index(m.group(2)) + 1} of {x}"
        else:
            new = s
        out.append({"start": seg["start"], "end": seg["end"], "sentence": new})
    return out


LABEL_VERSIONS = {
    "tgt": (relabel_tgt, "tgt_targetcarry", "official simple_subgoal boundaries; pick and place sentences carry their count k "
                                            "and the episode target x as ', k of x' (bean-scoop B9 label design); lowercased; "
                                            "built by cluster_robomme/build_pickxtimes_v2_labels.py --label-version tgt"),
    "v2": (relabel, "v2twophase", "official simple_subgoal boundaries (planner view); place segments carry the ordinal of "
                                  "their pick; all sentences lowercased; built by cluster_robomme/build_pickxtimes_v2_labels.py"),
    "official": (relabel_official, "official", "verbatim released simple_subgoal sentences and boundaries, lowercased; "
                                               "built by cluster_robomme/build_pickxtimes_v2_labels.py --label-version official"),
    "shift1": (relabel_shift1, "official_shift1", "COUNTERFACTUAL: official sentences with every pick ordinal moved one up "
                                                  "(five-pick episodes unchanged); oracle-write diagnostics only, never training"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-version", choices=tuple(LABEL_VERSIONS), default="v2")
    p.add_argument("--project-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    p.add_argument("--source-dataset", type=pathlib.Path,
                   default=pathlib.Path("/iris/u/kewalk/memory_v6_robomme/robomme/data/lerobot/PickXtimes"))
    p.add_argument("--skip-videos", action="store_true", help="metadata/parquet only (videos copied separately)")
    args = p.parse_args()
    root = args.project_root
    tag = args.label_version
    relabel_fn, version_name, policy = LABEL_VERSIONS[tag]
    meta_dir = root / "robomme/metadata/PickXtimes"
    manifest_path = meta_dir / "manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    legacy = json.loads((meta_dir / "subtasks.json").read_text())
    if legacy["source_manifest_sha256"] != manifest_sha:
        raise ValueError("legacy sidecar was derived from another manifest")
    lengths = {int(e["episode_index"]): int(e["expected_num_frames"]) for e in manifest["episodes"]}
    by_index = {int(e["episode_index"]): e for e in manifest["episodes"]}

    prompts = json.loads((args.source_dataset / "meta/episode_prompts.json").read_text())
    episodes, sentences = {}, set()
    for ep in sorted(by_index):
        rec = by_index[ep]
        official = legacy["episodes"][rec["stable_id"]]["segments"]
        segs = relabel_fn(official, rec["stable_id"])
        if tag == "tgt":  # the target in the sentences must be the count the goal asks for
            words = {"two": 2, "three": 3, "four": 4, "five": 5}
            prompt = prompts.get(str(ep), "")
            asked = next((n for w, n in words.items() if f"{w} times" in prompt), 1)
            written = int(re.search(r" of (\d)$", next(s["sentence"] for s in segs if s["sentence"].startswith("pick"))).group(1))
            if asked != written:
                raise ValueError(f"{rec['stable_id']}: prompt asks {asked} picks, labels have {written}")
        assert segs[0]["start"] == 0 and segs[-1]["end"] == lengths[ep] - 1
        assert all(a["end"] + 1 == b["start"] for a, b in zip(segs, segs[1:]))
        sentences.update(s["sentence"] for s in segs)
        episodes[rec["stable_id"]] = {"class": rec["class"], "num_frames": lengths[ep], "segments": segs,
                                      "official_segments": official}
    sentences = sorted(sentences)
    sidecar = {"schema_version": SIDECAR_SCHEMA, "dataset_version": f"PickXtimes_{version_name}",
               "source_manifest": "robomme/metadata/PickXtimes/manifest.json", "source_manifest_sha256": manifest_sha,
               "num_episodes": len(episodes), "sentences": sentences, "episodes": episodes,
               "label_policy": policy}
    sidecar["content_sha256"] = hashlib.sha256(
        (json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()).hexdigest()
    sidecar_sha = write_json(meta_dir / f"subtasks_{tag}.json", sidecar)
    print(f"sidecar: {len(episodes)} episodes, {len(sentences)} sentences, sha {sidecar_sha[:12]}", flush=True)

    tok = sentencepiece.SentencePieceProcessor(model_file=str(root / "v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    tokens = [tok.encode(s.lower().strip()) + tok.encode("\n") for s in sentences]
    assert max(map(len, tokens)) <= 48, "sentence exceeds the 48-token sentence buffer"

    # ---- LeRobot dataset with the v2 task column ----------------------------------------------------------------
    src, dst = args.source_dataset, root / f"robomme/data/lerobot/PickXtimes_{tag}"
    if (dst / "meta/info.json").exists():
        raise RuntimeError(f"{dst} exists; remove it before rebuilding")
    (dst / "meta").mkdir(parents=True)
    task_index = {s: i for i, s in enumerate(sentences)}
    with (dst / "meta/tasks.jsonl").open("w") as f:
        for s in sentences:
            f.write(json.dumps({"task_index": task_index[s], "task": s}) + "\n")
    info = json.loads((src / "meta/info.json").read_text())
    info["total_tasks"] = len(sentences)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    shutil.copy(src / "meta/episodes_stats.jsonl", dst / "meta/episodes_stats.jsonl")
    shutil.copy(src / "meta/episode_prompts.json", dst / "meta/episode_prompts.json")
    ep_lines = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines()]
    with (dst / "meta/episodes.jsonl").open("w") as f:
        for line in ep_lines:
            ep = int(line["episode_index"])
            segs = episodes[by_index[ep]["stable_id"]]["segments"]
            assert int(line["length"]) == lengths[ep]
            line["tasks"] = sorted({s["sentence"] for s in segs})
            f.write(json.dumps(line) + "\n")
    total = 0
    for ep in sorted(by_index):
        rel = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(src / rel)
        n = table.num_rows
        assert n == lengths[ep], (ep, n, lengths[ep])
        frame = table.column("frame_index").to_numpy()
        assert np.array_equal(frame, np.arange(n))
        segs = episodes[by_index[ep]["stable_id"]]["segments"]
        col = np.empty(n, dtype=np.int64)
        for s in segs:
            col[s["start"]: s["end"] + 1] = task_index[s["sentence"]]
        idx = table.schema.get_field_index("task_index")
        new = table.set_column(idx, table.schema.field(idx), pa.array(col, type=table.schema.field(idx).type))
        assert new.schema.metadata == table.schema.metadata  # HF features metadata preserved
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new, dst / rel)
        total += n
    assert total == info["total_frames"] == sum(lengths.values())
    print(f"parquet: {len(by_index)} episodes, {total} frames rewritten", flush=True)
    if not args.skip_videos:
        t0 = time.monotonic()
        shutil.copytree(src / "videos", dst / "videos")
        print(f"videos copied in {time.monotonic() - t0:.0f}s", flush=True)
    norm_sha = hashlib.sha256((root / "robomme/assets/PickXtimes/norm_stats.json").read_bytes()).hexdigest()
    spec = {"task": "PickXtimes", "label_version": version_name, "episodes": len(episodes), "frames": total,
            "manifest_sha256": manifest_sha, f"subtasks_{tag}_sha256": sidecar_sha, "norm_stats_sha256": norm_sha,
            "sentences": sentences, "reference_tokens": tokens, "max_sentence_tokens": max(map(len, tokens)),
            "max_segments_per_episode": max(len(e["segments"]) for e in episodes.values()),
            "lerobot_dataset": f"robomme/data/lerobot/PickXtimes_{tag}", "source_dataset": str(src),
            "state": "joint_state[0:7] + gripper_state[0:1] (finger width in meters)",
            "actions": "absolute joint_action[0:8], gripper -1 close/+1 open; observation row t supervises actions from t+1",
            "videos_copied": not args.skip_videos}
    write_json(meta_dir / f"prepared_{tag}.json", spec)
    print(f"PREPARED {tag}: {len(sentences)} sentences, max {spec['max_segments_per_episode']} segments/episode", flush=True)


if __name__ == "__main__":
    main()
