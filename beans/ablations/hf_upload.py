"""Push the beans0922 artefacts to the Hugging Face Hub (run once, from the machine that holds them).

  python beans/ablations/hf_upload.py dataset  [--src <lerobot dataset dir>]   # 89 episodes, ~49 GB, resumable
  python beans/ablations/hf_upload.py base --step 5000|10000 [--src <params dir>]   # the KI base checkpoint; replaces params/

Repos (public): datasets/kewalk123/yam_bean_scoop_0905_v5 and kewalk123/beans0922_pi05_base_10k. The dataset repo keeps the
LeRobot layout at its root (data/, meta/) plus the openpi norm stats under openpi_assets/ so 00_download.sh can restore both.
Needs a write token (huggingface-cli login) under $HOME/.cache/huggingface.
"""

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi

ROOT = pathlib.Path(os.environ.get("MEMORY_PROJECT_ROOT") or pathlib.Path(__file__).resolve().parents[2])
DATASET_REPO = "kewalk123/yam_bean_scoop_0905_v5"
BASE_REPO = "kewalk123/beans0922_pi05_base_10k"
NORM_STATS_REL = "v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json"
BASE_PARAMS_REL = "beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params"

DATASET_CARD = """---
license: cc-by-4.0
task_categories: [robotics]
tags: [lerobot, yam, bimanual, memory, openpi]
---
# yam/bean_scoop_0905_v5 -- LED bean scoop (real YAM station)

89 teleoperated episodes (71,089 frames at 30 Hz, 3 cameras 480x640, 14-D joint state/actions) of the task
"scoop the beans into the tray as many times as the green light blinked". The green LED blinks 1-3 times at the start of the
episode, then a yellow light signals "go"; the robot must remember the count while scooping. LeRobot v2 layout (`data/`, `meta/`), images stored in the parquet files.

Extras: `openpi_assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json` = the openpi normalisation
statistics of this dataset. Per-frame sub-task labels and the episode manifest live in the code repository
(github.com/ZJU-Walker/memory_project_0922, `openpi/cluster_v5/beans/`). Restore both with `beans/ablations/00_download.sh`.
"""

BASE_CARD = """---
license: apache-2.0
tags: [openpi, pi05, robotics, yam]
---
# beans0922 pi0.5 base (knowledge insulation), step {step}

The plain pi0.5 base of the beans0922 line: openpi `pi05_base` fine-tuned on `kewalk123/yam_bean_scoop_0905_v5` with the
knowledge-insulation recipe (sub-task sentence + FAST tokens supervise the language side, flow matching trains the action
expert under a stop-gradient prefix), batch 16, lr 5e-5, EMA 0.999 (config `pi05_yam_beans0922_base`,
github.com/ZJU-Walker/memory_project_0922). `params/` is the orbax checkpoint of update **{step}** (the file `STEP` says
which; the final one is 10000 -- an earlier step is published so training can start before the base run finishes), the one
the memory runs warm-start from (`OPENPI_BEANS_BASE_PARAMS`). Restore with `beans/ablations/00_download.sh`.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("dataset", "base"))
    ap.add_argument("--src", default=None, help="source directory (default: the tree's own copy)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step", type=int, default=None, help="base: the checkpoint step being pushed (default: from --src / the 10000 dir)")
    args = ap.parse_args()
    api = HfApi()
    print("hub user:", api.whoami()["name"], flush=True)
    if args.what == "dataset":
        src = pathlib.Path(args.src or ROOT / "v5/data/lerobot/yam/bean_scoop_0905_v5")
        if not (src / "meta/info.json").is_file():
            sys.exit(f"not a LeRobot dataset dir: {src}")
        api.create_repo(DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=DATASET_CARD.encode(), path_in_repo="README.md", repo_id=DATASET_REPO, repo_type="dataset")
        norm = ROOT / NORM_STATS_REL
        if norm.is_file():
            api.upload_file(path_or_fileobj=str(norm), path_in_repo="openpi_assets/" + NORM_STATS_REL.split("v5/assets/", 1)[1],
                            repo_id=DATASET_REPO, repo_type="dataset")
            print("norm stats uploaded", flush=True)
        print(f"uploading {src} -> {DATASET_REPO} ({args.workers} workers, resumable)", flush=True)
        api.upload_large_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(src), num_workers=args.workers,
                                allow_patterns=["data/**", "meta/**", "videos/**"], print_report_every=60)
    else:
        step = args.step
        if args.src is None and step is not None:
            src = ROOT / BASE_PARAMS_REL.replace("/10000/", f"/{step}/")
        else:
            src = pathlib.Path(args.src or ROOT / BASE_PARAMS_REL)
        if step is None:
            step = int(src.parent.name)  # .../<step>/params
        if not src.is_dir():
            sys.exit(f"base params dir missing: {src}")
        api.create_repo(BASE_REPO, repo_type="model", private=False, exist_ok=True)
        # replace, never merge: orbax chunk files are content-named, a merged params/ folder would mix two checkpoints
        try:
            api.delete_folder(path_in_repo="params", repo_id=BASE_REPO, repo_type="model", commit_message=f"replace params/ with step {step}")
            print("remote params/ removed", flush=True)
        except Exception as exc:  # noqa: BLE001 -- first push: nothing to delete
            print(f"(no remote params/ to remove: {type(exc).__name__})", flush=True)
        print(f"uploading {src} -> {BASE_REPO}/params (step {step}, {args.workers} workers, resumable)", flush=True)
        api.upload_large_folder(repo_id=BASE_REPO, repo_type="model", folder_path=str(src.parent), num_workers=args.workers,
                                allow_patterns=["params/**"], print_report_every=60)
        api.upload_file(path_or_fileobj=f"{step}\n".encode(), path_in_repo="STEP", repo_id=BASE_REPO, repo_type="model")
        api.upload_file(path_or_fileobj=BASE_CARD.format(step=step).encode(), path_in_repo="README.md", repo_id=BASE_REPO, repo_type="model")
        print(f"STEP={step} recorded", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
