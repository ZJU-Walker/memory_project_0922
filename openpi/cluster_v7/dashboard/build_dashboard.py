"""Build the boba training dashboard (static HTML with embedded data) from the v6/v7 training logs.
Usage: python3 build_dashboard.py [gpu_status_json] > boba_training_dashboard.html"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import sys

V7 = pathlib.Path("/iris/u/kewalk/memory_project_v7/v7")
V6 = pathlib.Path("/iris/u/kewalk/memory_project_v6/v6")
TODAY = dt.date(2026, 9, 12)
STEP_RE = re.compile(r"^Step (\d+): (.*)$")
PROG_RE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.\d+ \[I\] Progress on: ([\d.]+)it/(\d+)it rate:([\d.]+)s/it")
LAUNCH_RE = re.compile(r"^launch (\d\d)/(\d\d) (\d\d):(\d\d) .*config=(\S+) exp=(\S+) batch=(\d+) accum=(\d+) mode=(\S+) code=(\S+)")
EXIT_RE = re.compile(r"^exit=(\d+) (\d\d)/(\d\d) (\d\d):(\d\d)")


def parse_steps(path: pathlib.Path, keys: list[str]) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        row = {"step": int(m.group(1))}
        for kv in m.group(2).split(", "):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k in keys:
                try:
                    row[k.split("/")[-1]] = float(v)
                except ValueError:
                    pass
        rows.append(row)
    # keep the last occurrence per step (a relaunch appends to the same log)
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def parse_progress(path: pathlib.Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    day = TODAY
    prev = None
    for line in path.read_text(errors="replace").splitlines():
        m = PROG_RE.match(line)
        if not m:
            continue
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = dt.datetime.combine(day, dt.time(h, mi, s))
        if prev is not None and t < prev - dt.timedelta(hours=1):  # midnight wrap
            day = day + dt.timedelta(days=1)
            t = dt.datetime.combine(day, dt.time(h, mi, s))
        prev = t
        out.append({"t": t.isoformat(timespec="seconds"), "step": float(m.group(4)), "total": int(m.group(5))})
    return out


def parse_status(path: pathlib.Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        m = LAUNCH_RE.match(line)
        if m:
            events.append({"kind": "launch", "t": f"2026-{m.group(1)}-{m.group(2)}T{m.group(3)}:{m.group(4)}", "config": m.group(5),
                           "exp": m.group(6), "batch": int(m.group(7)), "accum": int(m.group(8)), "mode": m.group(9), "code": m.group(10)})
            continue
        m = EXIT_RE.match(line)
        if m:
            events.append({"kind": "exit", "code": int(m.group(1)), "t": f"2026-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}"})
    return events


def checkpoints(cfg: str, exp: str, root: pathlib.Path) -> list[int]:
    d = root / "checkpoints" / cfg / exp
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit())


MEM_KEYS = ["ce_loss", "flow_loss", "loss", "grad_norm", "memory_grad_norm", "param_norm",
            "diagnostic/v4_decision_ce_sum", "diagnostic/v4_decision_count", "diagnostic/v4_sem_commit_count",
            "diagnostic/v5_exact_decision_sum", "diagnostic/v5_exact_evidence_sum", "diagnostic/v5_evidence_count",
            "diagnostic/v5_token_acc_decision_sum", "diagnostic/v5_qk_cos_sum", "diagnostic/v5_qk_count",
            "diagnostic/v5_prefill_sentence_count", "diagnostic/v5_sentence_changed_count", "diagnostic/v5_write_requested_count",
            "diagnostic/v35_transition_count", "sequence_bucket_steps", "sequence_valid_fraction", "diagnostic/v4_sem_raw_read_rms_sum"]
BASE_KEYS = ["ce_loss", "flow_loss", "loss", "grad_norm"]

runs = {}
for tag, cfg, exp, root in [
    ("memA", "pi05_yam_mem_v7_bobaA", "v7_bobaA_20260912_r3", V7),
    ("memB", "pi05_yam_mem_v7_bobaB", "v7_bobaB_20260912_r3", V7),
]:
    log = root / "logs" / f"train_{exp}.log"
    runs[tag] = {"config": cfg, "exp": exp, "steps": parse_steps(log, MEM_KEYS), "progress": parse_progress(log),
                 "status": parse_status(root / "logs" / f"train_{exp}_status.log"), "ckpts": checkpoints(cfg, exp, root),
                 "total": 251 if tag == "memA" else 1501}
for tag in ("ctx_none", "ctx_prev", "ctx_state", "ctx_both"):
    exp = f"{tag}_20260912_r1"
    cfg = f"pi05_yam_boba0911_{tag}"
    log = V7 / "logs" / f"train_{exp}.log"
    runs[tag] = {"config": cfg, "exp": exp, "steps": parse_steps(log, BASE_KEYS), "progress": parse_progress(log),
                 "status": parse_status(V7 / "logs" / f"train_{exp}_status.log"), "ckpts": checkpoints(cfg, exp, V7), "total": 5000}
base_exp = "pi05_boba0911_base_rtc15_20260912_r1"
runs["base"] = {"config": "pi05_yam_boba0911_base", "exp": base_exp,
                "steps": parse_steps(V6 / "logs" / f"train_{base_exp}.log", BASE_KEYS), "progress": [],
                "status": parse_status(V6 / "logs" / f"train_{base_exp}_status.log"),
                "ckpts": checkpoints("pi05_yam_boba0911_base", base_exp, V6), "total": 10000}

def log_tail(path: pathlib.Path, n: int = 70, mem: bool = False) -> list[str]:
    """Compact, human-readable tail of a training log: abbreviated Step lines, progress lines, errors, loader/sampler info."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.rstrip()
        m = STEP_RE.match(line)
        if m:
            kv = {}
            for part in m.group(2).split(", "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        kv[k.split("/")[-1]] = float(v)
                    except ValueError:
                        pass
            txt = f"Step {m.group(1)}: ce {kv.get('ce_loss', float('nan')):.4f}  flow {kv.get('flow_loss', float('nan')):.4f}  loss {kv.get('loss', float('nan')):.4f}  grad {kv.get('grad_norm', float('nan')):.2f}"
            if mem:
                dc = kv.get("v4_decision_count") or 0
                if dc:
                    txt += f"  decision-exact {100 * kv.get('v5_exact_decision_sum', 0) / dc:.0f}%  decision-ce {kv.get('v4_decision_ce_sum', 0) / dc:.3f}"
                txt += f"  mem-grad {kv.get('memory_grad_norm', float('nan')):.3f}  commits/window {kv.get('v4_sem_commit_count', float('nan')):.1f}"
            out.append(txt)
            continue
        m = PROG_RE.match(line)
        if m:
            out.append(f"{m.group(1)}:{m.group(2)}:{m.group(3)}  update {m.group(4).rstrip('0').rstrip('.')}/{m.group(5)}  {m.group(6)} s/it")
            continue
        low = line.lower()
        if any(k in low for k in ("traceback", "error", "cancelled", "killed", "resource_exhausted", "nan")) and "constant folding" not in low and "this isn't necessarily" not in low and "xla_dump" not in low:
            out.append(line[:200])
        elif any(k in line for k in ("sequence sampling", "bucket sampling", "Audited partial initialization", "Initialized train state", "Saved checkpoint", "checkpoint")) and "[I]" in line:
            out.append(line.split("[I] ", 1)[-1].split(" (", 1)[0][:200] if "[I] " in line else line[:200])
    return out[-n:]


def compact_progress(prog: list[dict]) -> list[dict]:
    """Keep the last 120 progress lines dense and subsample the rest (document size)."""
    if len(prog) <= 400:
        return prog
    head, tail = prog[:-120], prog[-120:]
    step = max(1, len(head) // 280)
    return head[::step] + tail


for tag, run in runs.items():
    run["progress"] = compact_progress(run["progress"])

logs = {
    "memory": log_tail(V7 / "logs" / f"train_{runs['memB']['exp']}.log", mem=True) or log_tail(V7 / "logs" / f"train_{runs['memA']['exp']}.log", mem=True),
    "memory_run": runs["memB"]["exp"] if (V7 / "logs" / f"train_{runs['memB']['exp']}.log").exists() else runs["memA"]["exp"],
    "chain": (V7 / "logs" / "chain_mem_hgx2_4gpu.log").read_text().splitlines()[-12:] if (V7 / "logs" / "chain_mem_hgx2_4gpu.log").exists() else [],
}
ctx_running = None
for tag in ("ctx_none", "ctx_prev", "ctx_state", "ctx_both"):
    ev = runs[tag]["status"]
    if any(e["kind"] == "launch" for e in ev) and not any(e["kind"] == "exit" and e["t"] >= max(x["t"] for x in ev if x["kind"] == "launch") for e in ev):
        ctx_running = tag
logs["ctx_run"] = ctx_running or "ctx_both"
logs["ctx"] = log_tail(V7 / "logs" / f"train_{logs['ctx_run']}_20260912_r1.log", n=40)

gpu = {}
if len(sys.argv) > 1 and pathlib.Path(sys.argv[1]).exists():
    gpu = json.loads(pathlib.Path(sys.argv[1]).read_text())

data = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "runs": runs, "gpu": gpu, "logs": logs}
pathlib.Path(__file__).with_name("dash_data.json").write_text(json.dumps(data))
import base64
template = pathlib.Path(__file__).with_name("template.html").read_text()
page = template.replace("/*__DATA__*/", "const DATA = " + json.dumps(data) + ";")
# the copy the page publishes when the Refresh button asks the session for fresh data: same page, no nested copy, a marker
copy = page.replace("/*__SELF__*/", "const SELF_B64 = null;") + "\n<!-- refresh-request -->\n"
doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"></head><body>\n"
       + copy + "\n</body></html>\n")
self_b64 = base64.b64encode(doc.encode("utf-8")).decode("ascii")
sys.stdout.write(page.replace("/*__SELF__*/", "const SELF_B64 = \"" + self_b64 + "\";"))
