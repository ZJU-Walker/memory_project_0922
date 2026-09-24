"""Repository-scoped status, stop and recoverable archival; never cancel a GPU allocation."""
import argparse
import datetime
import os
from pathlib import Path
import re
import signal
import time

ROWS = ("snap", "snap_mlp3", "snap_mlp3_a9align", "snap_token_mlp3_a9align", "vis8", "vis8s", "state8", "vis8s_add", "state8_add")


def processes(root, row=None):
    """Only this UID + this checkout; do not print environment contents."""
    all_procs = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            cwd = (entry / "cwd").resolve()
            all_procs[int(entry.name)] = (int(fields[1]), fields[19], cmd, cwd)
        except (OSError, ValueError):
            continue
    protected = {os.getpid()}
    parent = os.getppid()
    while parent in all_procs and parent not in protected:
        protected.add(parent); parent = all_procs[parent][0]
    selected = set()
    for pid, (_, _, cmd, cwd) in all_procs.items():
        if pid in protected or "cluster_scripts/train_hs.py" in cmd:
            continue
        if not (cwd == root or root in cwd.parents):
            continue
        train = re.search(r"(?:^|/)train\.py\s+(pi05_yam_beans0922_[\w]+)(?:\s|$)", cmd)
        runner = re.search(r"(?:^|/)(run_(?:" + "|".join(ROWS) + r")\.sh|run_stages\.sh|train_slot_stage\.sh|train_ablation\.sh|train_beans0922_v4\.sh)(?:\s|$)", cmd)
        orphan = all_procs[pid][0] == 1 and "multiprocessing.spawn import spawn_main" in cmd
        if not (train or runner or orphan):
            continue
        # Row-scoped runners/workers need their exported CFG; never infer by a substring of the row name.
        try:
            env = dict(part.split(b"=", 1) for part in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in part)
        except OSError:
            env = {}
        config = train[1] if train else env.get(b"CFG", b"").decode(errors="replace")
        if row:
            expected = rf"pi05_yam_beans0922_ab_{re.escape(row)}(?:_A)?(?:_smoke)?"
            named_runner = runner and runner[1] == f"run_{row}.sh"
            if not re.fullmatch(expected, config) and not named_runner:
                continue
        elif orphan and not config.startswith("pi05_yam_beans0922_"):
            continue
        selected.add(pid)
    # Capture descendants before terminating launchers; includes dataloader workers.
    while True:
        children = {pid for pid, info in all_procs.items() if info[0] in selected and pid not in protected and "train_hs.py" not in info[2]}
        if children <= selected:
            break
        selected |= children
    return {pid: all_procs[pid] for pid in selected}


def same_process(pid, start):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] == start
    except (OSError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop", "archive"))
    parser.add_argument("row", nargs="?", choices=ROWS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--apply", action="store_true", help="actually move obsolete ab_<row>/smoke_ab_<row> directories")
    args = parser.parse_args(); root = args.root.resolve()
    found = processes(root, args.row)
    if args.action == "archive":
        if found:
            raise SystemExit("Stop this checkout's training before archiving.")
        dest = root / "beans/checkpoints/_archive" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for row in ((args.row,) if args.row else ROWS):
            for suffix, exp in (("", f"ab_{row}"), ("_smoke", f"smoke_ab_{row}"), ("", f"probe_{row}")):
                source = root / f"beans/checkpoints/pi05_yam_beans0922_ab_{row}{suffix}" / exp
                if source.is_dir() and not source.is_symlink():
                    target = dest / source.parent.name / source.name
                    print(f"{'MOVE' if args.apply else 'DRY RUN'} {source} -> {target}")
                    if args.apply:
                        target.parent.mkdir(parents=True, exist_ok=True); source.rename(target)
        print("Archive is recoverable; base weights, historical v4e runs, data and evaluation results are untouched.")
        return
    for pid, (_, _, cmd, _) in sorted(found.items()):
        print(pid, cmd[:240])
    print(f"{len(found)} matching processes; keep-alives and allocations are never stopped.")
    if args.action == "stop":
        # Root launchers first so a stage cannot relaunch after its child exits.
        order = sorted(found, key=lambda pid: found[pid][0] in found)
        for pid in order:
            if same_process(pid, found[pid][1]):
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
        time.sleep(3)
        for pid in order:
            if same_process(pid, found[pid][1]):
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass


if __name__ == "__main__":
    main()
