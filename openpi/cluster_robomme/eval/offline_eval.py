"""Offline memory eval on recorded expert episodes (record_expert_val.py): feed the recorded frames to the policy
server at the memory stride, with the model's OWN writes, through the same HTTP contract as the real rollouts, and
score the decoded sentence against the true sentence (official online label rewritten into the checkpoint's label
vocabulary). Actions are returned but never executed: the frames are the expert's, so this scores the writer/reader
under correct motion and says nothing about the arm.

    offline_eval.py --policy-url http://127.0.0.1:18770 --records <root>/robomme/expert_val --out <dir> [--chunk-size 15]
"""
import argparse
import json
import pathlib
import re
import time

import numpy as np

from common import HttpPolicyClient, jsonable, observation
from rollout import TgtLabelAdapter


def phase(s):
    s = (s or "").lower()
    return "pick" if s.startswith("pick") else "place" if s.startswith("place") else "press" if "press" in s else "done" if "completed" in s else "?"


def count_of(s):
    m = re.search(r", (\d+) of (\d+)$", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def score(rows):
    n = len(rows)
    exact = sum(r["subtask"] == r["true"] for r in rows)
    ph = sum(phase(r["subtask"]) == phase(r["true"]) for r in rows)
    model_flips = sum(1 for a, b in zip(rows, rows[1:]) if a["subtask"] != b["subtask"])
    true_changes = sum(1 for a, b in zip(rows, rows[1:]) if a["true"] != b["true"])
    first_wrong = next((r["frame"] for r in rows if r["subtask"] != r["true"]), None)
    # early flips: the model changes its sentence while the true sentence has not changed since the previous query
    early = sum(1 for a, b in zip(rows, rows[1:]) if a["subtask"] != b["subtask"] and a["true"] == b["true"])
    true_press = next((r["frame"] for r in rows if phase(r["true"]) == "press"), None)
    said_press = next((r["frame"] for r in rows if phase(r["subtask"]) == "press"), None)
    max_k_true = max((count_of(r["true"]) or (0, 0))[0] for r in rows)
    max_k_said = max((count_of(r["subtask"]) or (0, 0))[0] for r in rows)
    return dict(queries=n, exact=exact, exact_pct=round(100 * exact / n, 1), phase_pct=round(100 * ph / n, 1),
                model_flips=model_flips, true_changes=true_changes, early_flips=early, first_wrong_frame=first_wrong,
                true_press_frame=true_press, said_press_frame=said_press,
                press_verdict=("never" if said_press is None else "early" if true_press is None or said_press < true_press else "ok"),
                max_count_true=max_k_true, max_count_said=max_k_said, final_writes=rows[-1].get("writes"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--policy-url", required=True)
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--label-version", choices=("tgt", "official"), default="tgt")
    p.add_argument("--write-mode", choices=("own", "oracle"), default="own",
                   help="own = the model writes its own sentences (stage B test); oracle = the true sentence is written, the decode is scored (stage A test)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = HttpPolicyClient(args.policy_url, args.task, args.chunk_size)
    if bool(client.metadata.get("diagnostic_oracle_write")) != (args.write_mode == "oracle"):
        raise SystemExit("server/client write modes do not match (oracle writes need a --diagnostic-oracle-write server)")
    print(f"write mode: {args.write_mode} | checkpoint update {client.metadata.get('trained_update')}", flush=True)
    summary = []
    for d in sorted(args.records.glob(f"{args.task}_*_ep*")):
        meta = json.loads((d / "meta.json").read_text())
        z = np.load(d / "frames.npz")
        front, wrist, joint, grip = z["front"], z["wrist"], z["joint"], z["gripper"]
        goal, labels = meta["goal"], meta["labels"]
        adapter = TgtLabelAdapter(goal) if args.label_version == "tgt" else (lambda s: (s or "").strip().lower())
        true = [adapter(l) for l in labels]  # sequential: the place sentence takes the ordinal of the last pick
        client.reset(args.seed)
        rows = []
        for f in range(len(front)):
            ob = {"front_rgb_list": [front[f]], "wrist_rgb_list": [wrist[f]], "joint_state_list": [joint[f]], "gripper_state_list": [grip[f]]}
            client.observe(ob)
            if f % args.chunk_size:
                continue
            t0 = time.monotonic()
            pred = client.infer_oracle_write(observation(ob, goal), true[f]) if args.write_mode == "oracle" else client.infer(observation(ob, goal))
            rows.append(dict(frame=f, subtask=pred.get("subtask"), confidence=pred.get("subtask_confidence"), writes=pred.get("writes"),
                             bank=pred.get("bank"), true=true[f], true_official=labels[f], status=meta["status"][f],
                             latency_ms=round((time.monotonic() - t0) * 1000)))
        s = score(rows)
        s.update(episode=meta["episode"], split=meta["split"], goal=goal, frames=len(front), expert_status=meta["final_status"], dir=d.name, write_mode=args.write_mode)
        summary.append(s)
        (args.out / f"{d.name}.json").write_text(json.dumps(dict(meta={k: v for k, v in meta.items() if k not in ("labels", "grounded", "status")},
                                                                  metrics=s, rows=jsonable(rows)), indent=1))
        seq = " > ".join(f"{r['frame']}:{r['subtask']}" for a, r in zip([None] + rows, rows) if a is None or a["subtask"] != r["subtask"])
        print(f"ep{meta['episode']:>3} expert={meta['final_status']:7} queries={s['queries']:>3} exact={s['exact_pct']:5.1f}% phase={s['phase_pct']:5.1f}% "
              f"flips={s['model_flips']:>2}/{s['true_changes']} early={s['early_flips']:>2} press={s['press_verdict']:5} "
              f"count said/true={s['max_count_said']}/{s['max_count_true']} first-wrong={s['first_wrong_frame']}", flush=True)
        print("      " + seq[:400], flush=True)
    (args.out / "summary.json").write_text(json.dumps(dict(policy=client.metadata, chunk_size=args.chunk_size, seed=args.seed, write_mode=args.write_mode,
                                                            label_version=args.label_version, episodes=summary), indent=1))
    if summary:
        print(f"MEAN exact {np.mean([s['exact_pct'] for s in summary]):.1f}%  phase {np.mean([s['phase_pct'] for s in summary]):.1f}%  "
              f"press ok {sum(s['press_verdict'] == 'ok' for s in summary)}/{len(summary)}  early flips {sum(s['early_flips'] for s in summary)}", flush=True)


if __name__ == "__main__":
    main()
