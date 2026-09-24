#!/usr/bin/env python3
"""Upload one finalized Beans Orbax params checkpoint, without touching training.

Run with the project's openpi/.venv/bin/python. No GPU or model imports.
Private model repository by default; --public is an explicit publishing choice.
Only params/, its completion metadata, and a small provenance manifest are sent.
"""

import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings


PARAM_FILE = re.compile(
    r"(?:_METADATA|_sharding|manifest\.ocdbt|array_metadatas/process_\d+|"
    r"d/[0-9a-f]+|ocdbt\.process_\d+/(?:manifest\.ocdbt|d/[0-9a-f]+))"
)


def git_value(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if checkpoint.name == "params":
        checkpoint = checkpoint.parent
    if not checkpoint.name.isdecimal():
        raise ValueError("Select a finalized numeric step directory, not an Orbax temporary directory.")
    config, experiment = checkpoint.parent.parent.name, checkpoint.parent.name
    for name in (config, experiment):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unexpected config/experiment directory name.")
    if not config.startswith("pi05_yam_beans0922_"):
        raise ValueError("Expected .../pi05_yam_beans0922_<config>/<experiment>/<step>.")
    marker = checkpoint / "_CHECKPOINT_METADATA"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Missing checkpoint completion metadata.")
    metadata = json.loads(marker.read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError("Checkpoint has no commit timestamp; wait for saving to finish.")
    params = checkpoint / "params"
    if params.is_symlink() or not params.is_dir():
        raise ValueError("Expected a real params/ directory.")
    files = {}
    for path in sorted(params.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink inside params/: {path.name}")
        if path.is_dir():
            continue
        relative = path.relative_to(params).as_posix()
        if not path.is_file() or not PARAM_FILE.fullmatch(relative):
            raise ValueError(f"Unexpected file in params/ (not uploaded): {relative}")
        files[f"params/{relative}"] = path
    for required in ("params/_METADATA", "params/manifest.ocdbt"):
        if required not in files:
            raise ValueError(f"Incomplete Orbax checkpoint: missing {required}")
    if not any("/d/" in name for name in files):
        raise ValueError("No Orbax parameter data chunks found.")
    files["_CHECKPOINT_METADATA"] = marker
    sizes = {name: path.stat().st_size for name, path in files.items()}
    prefix = f"checkpoints/{config}/{experiment}/{checkpoint.name}"
    return checkpoint, config, experiment, prefix, files, sizes


def build_manifest(root, checkpoint, config, experiment, sizes):
    head = git_value(root, "rev-parse", "HEAD")
    dirty = git_value(root, "status", "--porcelain", "--untracked-files=no")
    if head is None:
        raise ValueError("--repo-root must point to the training Git checkout.")
    launch_code = None
    log = root / "beans/ablations/logs" / f"train_{experiment}.log"
    if log.is_file():
        with log.open(errors="replace") as stream:
            for _ in range(300):
                line = stream.readline()
                if not line:
                    break
                match = re.search(r"^launch .*\bcode=([0-9a-f]{7,40})\b", line)
                if match:
                    launch_code = match.group(1)
                    break
    recipe = {}
    recipe_path = root / "beans/ablations/logs" / f"{experiment}.recipe"
    if recipe_path.is_file():
        text = recipe_path.read_text()
        for name in ("row", "stage", "batch", "accum", "steps", "A", "prefill"):
            match = re.search(rf"(?:^|\s){name}=([A-Za-z0-9_-]+)(?:\s|$)", text)
            if match:
                recipe[name] = match.group(1)
    return {
        "schema": "beans0922.eval_transfer.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config,
        "experiment_name": experiment,
        "checkpoint_step": int(checkpoint.name),
        "format": "orbax-ocdbt-params",
        "purpose": "Evaluation only; optimizer state and dataset are not included.",
        "code_repository": "https://github.com/ZJU-Walker/memory_project_0922",
        "git_head_at_upload": head,
        "tracked_changes_at_upload": bool(dirty),
        "first_launch_code_from_log": launch_code,
        "recipe": recipe,
        "files": [{"path": name, "bytes": size} for name, size in sorted(sizes.items())],
    }


def publish(api, operation_class, repo_id, prefix, files, sizes, manifest, public=False):
    api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="model")
    if not info.private and not public:
        raise ValueError("This repository is public. Use a private repository, or explicitly pass --public.")
    parent = info.sha
    existing = api.list_repo_files(repo_id, repo_type="model", revision=parent)
    if any(name == prefix or name.startswith(prefix + "/") for name in existing):
        raise ValueError("This checkpoint destination already exists; nothing overwritten. Use its existing revision or another repo.")
    # Parent-pinned atomic commit: params and the completion manifest appear together.
    # Exact file allowlist, no folder upload, no delete operations, no local staging copy.
    for name, path in files.items():
        if path.stat().st_size != sizes[name]:
            raise ValueError("Checkpoint changed after inspection; retry with a retained, completed step.")
    operations = [operation_class(path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(path))
                  for name, path in files.items()]
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    operations.append(operation_class(
        path_in_repo=f"{prefix}/eval_manifest.json",
        path_or_fileobj=manifest_bytes,
    ))
    commit = api.create_commit(
        repo_id=repo_id, repo_type="model", operations=operations,
        parent_commit=parent, num_threads=2,
        commit_message=f"Add eval checkpoint {manifest['experiment_name']} step {manifest['checkpoint_step']}",
    )
    expected = {f"{prefix}/{name}": size for name, size in sizes.items()}
    expected[f"{prefix}/eval_manifest.json"] = len(manifest_bytes)
    uploaded = api.repo_info(repo_id, repo_type="model", revision=commit.oid, files_metadata=True)
    actual = {entry.rfilename: entry.size for entry in uploaded.siblings}
    if any(actual.get(name) != size for name, size in expected.items()):
        raise RuntimeError(f"Post-upload size verification failed at revision {commit.oid}; do not evaluate it yet.")
    return commit.oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help=".../<config>/<experiment>/1000 (or its params/)")
    parser.add_argument("--repo-id", required=True, help="Your HF username/beans0922-eval")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Training checkout, default current directory")
    parser.add_argument("--public", action="store_true", help="Explicitly permit publishing model weights publicly")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect local metadata; no login, upload, or GPU")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id):
        parser.error("--repo-id must be HF_USERNAME/REPOSITORY")
    checkpoint, config, experiment, prefix, files, sizes = inspect_checkpoint(args.checkpoint)
    manifest = build_manifest(args.repo_root.resolve(), checkpoint, config, experiment, sizes)
    print(f"Checkpoint: {config}/{experiment}/{checkpoint.name}", flush=True)
    print(f"Upload: {len(files)} files, {sum(sizes.values()) / 2**30:.2f} GiB (params only; no optimizer/data/logs)", flush=True)
    print(f"Destination: {args.repo_id}/{prefix} ({'PUBLIC' if args.public else 'private required'})", flush=True)
    print(f"Code: launch={manifest['first_launch_code_from_log']} upload_checkout={manifest['git_head_at_upload']}", flush=True)
    if manifest["tracked_changes_at_upload"]:
        print("WARNING: checkout has tracked changes; source edits are NOT included in this upload.", flush=True)
    if args.dry_run:
        print("DRY RUN OK: no files changed and no network request made.")
        return 0
    from huggingface_hub import CommitOperationAdd, HfApi, get_token
    token = get_token()
    if not token:
        if not sys.stdin.isatty():
            raise ValueError("Set HF_TOKEN in this shell or run interactively for hidden token input.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            token = getpass.getpass("Hugging Face write token (hidden): ").strip()
    if not token:
        raise ValueError("No Hugging Face token provided.")
    api = HfApi(endpoint="https://huggingface.co", token=token)
    api.whoami()  # Fail before any writes if the token is invalid. Never print the token.
    revision = publish(api, CommitOperationAdd, args.repo_id, prefix, files, sizes, manifest, args.public)
    print("\nUPLOAD_OK — paste the following three lines into the chat:", flush=True)
    print(f"HF_REPO={args.repo_id}\nREVISION={revision}\nSUBDIR={prefix}", flush=True)
    print(f"https://huggingface.co/{args.repo_id}/tree/{revision}/{prefix}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted. Training is untouched; rerun to retry the upload.")
    except Exception as exc:
        # Do not emit arbitrary library exception text: auth errors can contain sensitive URLs.
        if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)):
            sys.exit(f"ERROR: {exc}")
        sys.exit(f"Upload failed ({type(exc).__name__}); check network/write permissions and retry. No training was stopped.")
