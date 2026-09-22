"""Prompt-swap probe (09-17 20:00, user): does the FIRST decoded sentence's target count follow the prompt?
For every recorded expert scene, take frame 0 (empty bank after a reset) and ask the normal policy server once per
count word (one-pick prompt form for 1, "repeating this action <word> times" for 2..5). Diagnostic only.

    prompt_swap_probe.py --policy-url http://127.0.0.1:PORT --records <root>/robomme/expert_val --out <file.json>
"""
import argparse
import json
import pathlib
import re

import numpy as np

from common import HttpPolicyClient, observation

WORDS = {1: None, 2: "two", 3: "three", 4: "four", 5: "five"}


def prompt_for(goal, x):
    """Rewrite the benchmark goal for target count x, keeping the colour and the rest of the wording."""
    m = re.match(r"^pick up the (\w+) cube and place it on the target(?:, repeating this action \w+ times)?, then press the button to stop$", goal.strip())
    if not m:
        raise ValueError(f"unexpected goal form: {goal!r}")
    colour = m.group(1)
    if x == 1:
        return f"pick up the {colour} cube and place it on the target, then press the button to stop"
    return f"pick up the {colour} cube and place it on the target, repeating this action {WORDS[x]} times, then press the button to stop"


def said_x(s):
    m = re.search(r" of (\d+)$", s or "")
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-url", required=True)
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames", default="0", help="comma list of frame indices to probe (each with a fresh, empty bank)")
    args = p.parse_args()
    client = HttpPolicyClient(args.policy_url, args.task, args.chunk_size)
    print(f"checkpoint {client.metadata.get('training_config')} update {client.metadata.get('trained_update')}", flush=True)
    results = []
    for d in sorted(args.records.glob(f"{args.task}_*_ep*")):
        meta = json.loads((d / "meta.json").read_text()); z = np.load(d / "frames.npz")
        true_x = next((v for w, v in {"two": 2, "three": 3, "four": 4, "five": 5}.items() if f"{w} times" in meta["goal"]), 1)
        for f in [int(x) for x in args.frames.split(",")]:
            ob = {"front_rgb_list": [z["front"][f]], "wrist_rgb_list": [z["wrist"][f]], "joint_state_list": [z["joint"][f]], "gripper_state_list": [z["gripper"][f]]}
            row = {"episode": meta["episode"], "true_x": true_x, "frame": f, "said": {}}
            for x in (1, 2, 3, 4, 5):
                client.reset(args.seed)  # empty bank, as at the start of every episode
                client.observe(ob)
                pred = client.infer(observation(ob, prompt_for(meta["goal"], x)))
                row["said"][x] = {"sentence": pred.get("subtask"), "x": said_x(pred.get("subtask")), "conf": pred.get("subtask_confidence")}
            results.append(row)
            cells = " ".join(f"{x}->{row['said'][x]['x']}{'' if row['said'][x]['x'] == x else '!'}" for x in (1, 2, 3, 4, 5))
            print(f"scene ep{meta['episode']:>2} (true x={true_x}) frame {f:>3}: prompt count -> decoded x: {cells}", flush=True)
    follows = sum(1 for r in results for x in (1, 2, 3, 4, 5) if r["said"][x]["x"] == x)
    total = 5 * len(results)
    print(f"decoded x follows the prompt in {follows}/{total} probes", flush=True)
    args.out.write_text(json.dumps({"policy": client.metadata, "results": results, "follows": follows, "total": total}, indent=1))


if __name__ == "__main__":
    main()
