"""Counterfactual-history sidecar for the oracle-write pass (09-15, the "shifted prefilled history" test).

Copies subtasks_official.json and moves every pick ordinal one up (first -> second, ..., fourth -> fifth) in every
episode with at most four picks; episodes with five picks are left unchanged and listed in "unshifted_episodes".
Boundaries, place/press/completed sentences and the prompt are untouched. Feeding this sidecar to
`v5_heldout_video.py --write-mode oracle` writes a bank that claims one more pick than the video shows, while the
frames and the prompt are the true ones. A model whose ordinal follows the BANK then predicts the shifted ordinal
(and "press the button to stop" one cycle early, when the shifted count reaches the prompt's repeat count); a model
that ignores the bank predicts the true ordinal. Score the run's per-step predictions against both sidecars.
"""
import argparse
import hashlib
import json
import pathlib
import re

ORDINALS = ("first", "second", "third", "fourth", "fifth")
PICK = re.compile(r"^pick up the (\w+) cube for the (\w+) time$")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[2]
    p.add_argument("--source", type=pathlib.Path, default=root / "robomme/metadata/PickXtimes/subtasks_official.json")
    p.add_argument("--output", type=pathlib.Path, default=root / "robomme/metadata/PickXtimes/subtasks_official_shift1.json")
    args = p.parse_args()
    sidecar = json.loads(args.source.read_text())
    unshifted, shifted, sentences = [], 0, set()
    for sid, ep in sidecar["episodes"].items():
        picks = [s for s in ep["segments"] if PICK.match(s["sentence"])]
        if any(PICK.match(s["sentence"]).group(2) == ORDINALS[-1] for s in picks):
            unshifted.append(sid)
            sentences.update(s["sentence"] for s in ep["segments"])
            continue
        for seg in ep["segments"]:
            if m := PICK.match(seg["sentence"]):
                seg["sentence"] = f"pick up the {m.group(1)} cube for the {ORDINALS[ORDINALS.index(m.group(2)) + 1]} time"
        shifted += 1
        sentences.update(s["sentence"] for s in ep["segments"])
    sidecar["dataset_version"] = "PickXtimes_official_shift1"
    sidecar["label_policy"] = ("COUNTERFACTUAL: official sentences with every pick ordinal moved one up (episodes with five "
                               "picks unchanged); oracle-write diagnostics only, never training")
    sidecar["unshifted_episodes"] = unshifted
    sidecar["sentences"] = sorted(sentences)
    sidecar.pop("content_sha256", None)
    body = json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    sidecar["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.output.write_text(json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.output.name}: {shifted} episodes shifted, {len(unshifted)} left unchanged (five picks): {unshifted[:6]}...")


if __name__ == "__main__":
    main()
