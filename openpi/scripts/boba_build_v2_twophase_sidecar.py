"""Two-phase scoop labels for boba_0911 (2026-09-13, cluster_v7/README.md §3 "B/500 analysis"):
every `<drink>, scoop, k of n` segment of the v1 sidecar is cut at the POUR ONSET (the label builder's own detector:
right wrist rj4 > POUR_RJ4 after a dig) into `<drink>, scoop, k of n` (spoon arrives at the bin -> tilt onset) and
`<drink>, pour, k of n` (tilt onset -> next bin arrival / put the scoop back). Everything else is copied verbatim.

    .venv/bin/python scripts/boba_build_v2_twophase_sidecar.py \
        --v1 cluster_v7/boba/boba_v5_subtask_labels_v1.json --manifest cluster_v7/boba/boba_episode_manifest_v1.json \
        --out cluster_v7/boba/boba_v5_subtask_labels_v2.json
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import boba_build_subtask_labels as B  # POUR_RJ4 / DIG_RJ4 / pours() / runs()

SCOOP_RE = re.compile(r"^(first|second), scoop, (\d) of (\d)$")


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def split_episode(segments: list[dict], rj: np.ndarray, notes: list[str], sid: str) -> list[dict]:
    out = []
    for seg in segments:
        m = SCOOP_RE.match(seg["sentence"])
        if not m:
            out.append(dict(seg)); continue
        drink, k, n = m.groups()
        a, b = int(seg["start"]), int(seg["end"])
        pr = B.pours(rj, a, b)
        if len(pr) != 1:
            w = rj[a:b + 1, 4]; ex = B.runs(w > B.POUR_RJ4, 8)
            notes.append(f"{sid} {seg['sentence']}: {len(pr)} dig-preceded pours in [{a},{b}], excursions {ex}; using the last excursion")
            pr = [(a + s, a + e) for s, e in ex[-1:]]
        if not pr:
            raise SystemExit(f"{sid} {seg['sentence']}: no pour excursion inside the segment")
        p0 = int(pr[0][0])
        if not (a + 15 <= p0 <= b - 15):
            raise SystemExit(f"{sid} {seg['sentence']}: pour onset {p0} leaves a phase shorter than 0.5 s in [{a},{b}]")
        out.append({"start": a, "end": p0 - 1, "sentence": f"{drink}, scoop, {k} of {n}"})
        out.append({"start": p0, "end": b, "sentence": f"{drink}, pour, {k} of {n}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=pathlib.Path, required=True); ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--dataset-name", default="0911_boba_v2twophase")
    args = ap.parse_args()
    v1 = json.loads(args.v1.read_text()); manifest = json.loads(args.manifest.read_text())
    raw_dir = {e["stable_id"]: e["raw_dir"] for e in manifest["episodes"]}
    episodes, vocab, notes, lens = {}, set(), [], []
    for sid, rec in v1["episodes"].items():
        rj = np.load(pathlib.Path(raw_dir[sid]) / "right_joint_positions.npy")
        segs = split_episode(rec["segments"], rj, notes, sid)
        # tiling check: contiguous, no gaps, same span as v1
        prev_end = -1
        for s in segs:
            if s["start"] != prev_end + 1: raise SystemExit(f"{sid}: gap/overlap at {s}")
            prev_end = s["end"]; vocab.add(s["sentence"]); lens.append((s["end"] - s["start"] + 1, s["sentence"], sid))
        if prev_end != rec["segments"][-1]["end"]: raise SystemExit(f"{sid}: span changed")
        seq = [s["sentence"] for s in segs if ", scoop, " in s["sentence"] or ", pour, " in s["sentence"]]
        for i in range(0, len(seq), 2):  # scoop k must be followed by pour k
            if seq[i].replace("scoop", "pour") != seq[i + 1]: raise SystemExit(f"{sid}: bad alternation {seq}")
        new = dict(rec); new["segments"] = segs; episodes[sid] = new
    sidecar = {k: v for k, v in v1.items() if k not in ("content_sha256", "episodes", "sentences", "num_episodes", "dataset_version")}
    sidecar.update({"dataset_version": args.dataset_name, "num_episodes": len(episodes), "sentences": sorted(vocab), "episodes": episodes})
    body = _canonical(dict(sidecar)); sidecar["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    text = _canonical(sidecar); args.out.write_text(text, encoding="utf-8")
    shortest = sorted(lens)[:5]
    print(f"wrote {args.out}\n  file sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}\n  {len(vocab)} sentences: {sorted(vocab)}")
    print(f"  shortest phases (frames, sentence, episode): {shortest}")
    print(f"  pour phases: {sum(1 for L,s,_ in lens if ', pour, ' in s)}, scoop phases: {sum(1 for L,s,_ in lens if ', scoop, ' in s)}")
    print("  notes:" if notes else "  no fallbacks needed"); [print("   ", n) for n in notes]


if __name__ == "__main__":
    main()
