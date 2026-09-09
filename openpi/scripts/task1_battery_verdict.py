"""Summarise one task1 development battery directory (v6/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.json).

Prints one row per (mode, episode) and a final line `GATE PASS|FAIL: oracle first decision correct on X/6, self Y/6`
(PASS = oracle X >= --min-oracle, default 5). Missing summaries count as failures."""

import argparse
import json
import pathlib

DEV = (12, 13, 26, 27, 67, 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--min-oracle", type=int, default=5)
    args = parser.parse_args()
    rows, counts = [], {"oracle": 0, "self": 0}
    for mode in ("oracle", "self"):
        for ep in DEV:
            f = args.out_dir / f"ep{ep:02d}_{mode}.json"
            if not f.exists():
                rows.append(f"  {mode:6s} ep{ep:02d}: MISSING (see ep{ep:02d}_{mode}_run.log)")
                continue
            s = json.loads(f.read_text())
            ok = bool(s.get("first_decision_correct"))
            counts[mode] += int(ok)
            bank = [b for b in s.get("final_bank", [])]
            rows.append(
                f"  {mode:6s} ep{ep:02d} {s['stable_id'].split('/')[-1]:14s} prompt={s['prompt']!r:20s} "
                f"first={s.get('first_decision_pred')!r:14s} {'OK ' if ok else 'BAD'} "
                f"decisions {s['decision_side_correct']}/{s['decision_steps']} "
                f"evidence {s['evidence_pred_exact']}/{s['evidence_steps']} writes={s['writes']} bank={bank}"
            )
    print("\n".join(rows))
    verdict = "PASS" if counts["oracle"] >= args.min_oracle else "FAIL"
    print(f"GATE {verdict}: oracle first decision correct on {counts['oracle']}/6, self {counts['self']}/6")


if __name__ == "__main__":
    main()
