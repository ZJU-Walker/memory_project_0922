"""Summarise one task1 development battery directory (v6/diagnostics/videos_<exp>_<step>/ep<idx>_<mode>.json).

Prints one row per (mode, episode) and a final line `GATE PASS|FAIL: oracle first decision correct on X/6, self Y/6`
(PASS = oracle X >= --min-oracle, default 5). Missing summaries count as failures."""

import argparse
import json
import pathlib

DEV = (12, 13, 26, 27, 67, 68)
MODES = ("oracle", "oracle_evidence", "self")


def _recall(records: list[dict], closing_start: int | None = None) -> dict:
    """The closing segment = the restated note (`<target> in bin k`) just before the first decision step. Its FIRST
    step is the recall test proper: in oracle mode the closing note is not yet in the bank at that step; in
    oracle_evidence/self modes it is never handed over. `closing_start` (the sidecar's frame) is needed when the
    target was the LAST object placed: its placement note and the closing note are the same text back to back, and
    the run of identical labels would otherwise start at the placement (2026-09-09 02:35, demo19 box)."""
    i_dec = next((i for i, r in enumerate(records) if str(r["gt_now"]).startswith("open bin")), None)
    if i_dec is None or i_dec == 0:
        return {"label": None, "first_pred": None, "first_ok": False, "exact": 0, "steps": 0}
    label = records[i_dec - 1]["gt_now"]
    i0 = i_dec - 1
    while i0 > 0 and records[i0 - 1]["gt_now"] == label:
        i0 -= 1
    if closing_start is not None:
        i0 = max(i0, next((i for i, r in enumerate(records) if int(r["frame"]) >= closing_start), i0))
    seg = records[i0:i_dec]
    return {"label": label, "first_pred": seg[0]["pred"], "first_ok": seg[0]["pred"] == label,
            "exact": sum(1 for r in seg if r["pred"] == label), "steps": len(seg)}


def _decisions(records: list[dict], decision_start: int | None) -> dict:
    """Decision metrics from the SIDECAR boundary (lead30 line, 2026-09-09 12:15): the battery's own `decision` flag
    comes from the LeRobot task labels, whose decision segment still starts at the first joint motion. Decision steps
    = frames >= the sidecar's last-segment start; `still` = those before the first flagged (motion) step, i.e. the
    steps where the bank is the only source of the bin. With a v1 sidecar both boundaries coincide."""
    if decision_start is None:
        dec = [r for r in records if r.get("decision")]
    else:
        dec = [r for r in records if int(r["frame"]) >= decision_start]
    i_motion = next((i for i, r in enumerate(dec) if r.get("decision")), len(dec))
    still = dec[:i_motion]
    first = dec[0] if dec else None
    return {"first_pred": first["pred"] if first else None,
            "first_ok": bool(first) and first["pred"] == first["gt_target"],
            "exact": sum(1 for r in dec if r["pred"] == r["gt_target"]), "steps": len(dec),
            "still_exact": sum(1 for r in still if r["pred"] == r["gt_target"]), "still_steps": len(still)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--min-oracle", type=int, default=5)
    parser.add_argument("--sidecar", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "cluster_v6/task1/task1v6_v5_subtask_labels_v1.json",
                        help="v5 sentence sidecar (closing-segment frames)")
    args = parser.parse_args()
    sidecar = json.loads(args.sidecar.read_text()) if args.sidecar.exists() else None
    rows, counts, recalls = [], {m: 0 for m in MODES}, {m: 0 for m in MODES}
    for mode in MODES:
        for ep in DEV:
            f = args.out_dir / f"ep{ep:02d}_{mode}.json"
            if not f.exists():
                rows.append(f"  {mode:6s} ep{ep:02d}: MISSING (see ep{ep:02d}_{mode}_run.log)")
                continue
            s = json.loads(f.read_text())
            bank = [b for b in s.get("final_bank", [])]
            closing_start = decision_start = None
            if sidecar is not None and s["stable_id"] in sidecar.get("episodes", {}):
                segs = sidecar["episodes"][s["stable_id"]]["segments"]
                closing_start = int(segs[-2]["start"]) if len(segs) >= 2 else None
                decision_start = int(segs[-1]["start"])
            dec = _decisions(s["records"], decision_start)
            ok = dec["first_ok"]
            counts[mode] += int(ok)
            recall = _recall(s["records"], closing_start)
            recalls[mode] += int(recall["first_ok"])
            rows.append(
                f"  {mode:15s} ep{ep:02d} {s['stable_id'].split('/')[-1]:14s} prompt={s['prompt']!r:20s} "
                f"first={dec['first_pred']!r:14s} {'OK ' if ok else 'BAD'} "
                f"decisions {dec['exact']}/{dec['steps']} (still {dec['still_exact']}/{dec['still_steps']}) | recall {recall['label']!r}: first step said "
                f"{recall['first_pred']!r} ({'RIGHT' if recall['first_ok'] else 'wrong'}), segment {recall['exact']}/{recall['steps']} | "
                f"evidence {s['evidence_pred_exact']}/{s['evidence_steps']} writes={s['writes']} bank={bank}"
            )
    print("\n".join(rows))
    verdict = "PASS" if counts["oracle"] >= args.min_oracle else "FAIL"
    print(f"GATE {verdict}: first decision (sidecar boundary) correct — oracle {counts['oracle']}/6, oracle_evidence {counts['oracle_evidence']}/6, "
          f"self {counts['self']}/6; RECALL (first closing step) — oracle {recalls['oracle']}/6, "
          f"oracle_evidence {recalls['oracle_evidence']}/6, self {recalls['self']}/6")


if __name__ == "__main__":
    main()
