"""Stability report of the ablation probes: one table per row from the training log (Step lines) and, when reachable, the
W&B history (memory-group gradient norm, sensory-bank telemetry). Flags non-finite values, gradient-norm growth and a bank
norm that keeps growing instead of levelling off.

  python beans/ablations/probe_report.py [--rows vis8s_add state8_add vis8 snap] [--project beans0922_ablation] [--no-wandb]
"""

import argparse
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEP_RE = re.compile(r"Step (\d+): (.*)")


def parse_log(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        rec = {"step": int(m.group(1))}
        for kv in m.group(2).split(","):
            if "=" in kv:
                k, v = kv.strip().split("=", 1)
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
        rows.append(rec)
    return rows


def wandb_history(project: str, exp: str) -> dict[int, dict]:
    try:
        import wandb
    except ImportError:
        return {}
    keys = ["memory_grad_norm", "grad_norm", "vis_commit_rate", "vis_bank_norm", "vis_read_rms", "vis_read_injected_rms",
            "diagnostic/v4_sem_commit_count", "diagnostic/v4_sem_injected_pre_cast_rms_sum"]  # sentence-bank keys only with log_diagnostics
    api = wandb.Api(timeout=30)
    out: dict[int, dict] = {}
    try:
        runs = [r for r in api.runs(f"{api.default_entity}/{project}") if r.name == exp or r.config.get("exp_name") == exp]
    except Exception as exc:  # noqa: BLE001
        print(f"  (wandb unavailable: {exc})")
        return {}
    for run in runs[:1]:
        for rec in run.scan_history(keys=["_step", *keys]):
            step = rec.get("_step")
            if step is None:
                continue
            out[int(step)] = {k: rec.get(k) for k in keys if rec.get(k) is not None}
    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "!!" + str(v)
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def report(row: str, project: str, use_wandb: bool) -> list[str]:
    exp = f"probe_{row}"
    log = ROOT / "beans/ablations/logs" / f"train_{exp}.log"
    flags = []
    if not log.is_file():
        return [f"{row}: no log at {log}"]
    steps = parse_log(log)
    hist = wandb_history(project, exp) if use_wandb else {}
    print(f"\n== {row}  ({log.name}, {len(steps)} logged steps)")
    cols = ["step", "loss", "ce_loss", "flow_loss", "grad_norm", "memory_grad_norm", "vis_commit_rate", "vis_bank_norm",
            "vis_read_rms", "vis_inj_rms", "sem_commit"]
    print(" | ".join(f"{c:>14}" for c in cols))
    for rec in steps:
        h = hist.get(rec["step"], {})
        vals = [rec["step"], rec.get("loss"), rec.get("ce_loss"), rec.get("flow_loss"), rec.get("grad_norm"),
                rec.get("memory_grad_norm", h.get("memory_grad_norm")), rec.get("vis_commit_rate", h.get("vis_commit_rate")),
                rec.get("vis_bank_norm", h.get("vis_bank_norm")), rec.get("vis_read_rms", h.get("vis_read_rms")),
                rec.get("vis_read_injected_rms", h.get("vis_read_injected_rms")), h.get("diagnostic/v4_sem_commit_count")]
        print(" | ".join(f"{fmt(v):>14}" for v in vals))
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                flags.append(f"{row}: non-finite {k} at step {rec['step']}")
    g = [r["grad_norm"] for r in steps if "grad_norm" in r]
    if len(g) >= 4:
        first, last = sum(g[:2]) / 2, sum(g[-2:]) / 2
        if last > 3 * first and last > 10:
            flags.append(f"{row}: grad_norm grows {first:.1f} -> {last:.1f}")
        if max(g) > 1000:
            flags.append(f"{row}: grad_norm spike {max(g):.0f}")
    bank = [r["vis_bank_norm"] for r in steps if "vis_bank_norm" in r] or [h["vis_bank_norm"] for _, h in sorted(hist.items()) if "vis_bank_norm" in h]
    if len(bank) >= 6 and bank[-1] > 2 * bank[len(bank) // 2] and bank[-1] > bank[-2] > bank[-3]:
        flags.append(f"{row}: vis_bank_norm still growing at the end ({bank[len(bank)//2]:.3g} -> {bank[-1]:.3g})")
    losses = [r["loss"] for r in steps if "loss" in r]
    if len(losses) >= 4 and losses[-1] > 1.5 * losses[0]:
        flags.append(f"{row}: loss rises {losses[0]:.3g} -> {losses[-1]:.3g}")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=["vis8s_add", "state8_add", "vis8", "snap"])
    ap.add_argument("--project", default="beans0922_ablation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    flags = []
    for row in args.rows:
        flags += report(row, args.project, not args.no_wandb)
    print("\n== flags:" if flags else "\n== flags: none")
    for f in flags:
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
