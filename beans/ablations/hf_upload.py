"""Push the beans0922 artefacts to the Hugging Face Hub (run once, from the machine that holds them).

  python beans/ablations/hf_upload.py dataset  [--src <lerobot dataset dir>]   # 89 episodes, ~49 GB, resumable
  python beans/ablations/hf_upload.py base     [--src <base params dir>]       # the KI base checkpoint (step 10000)

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
"scoop the beans into the tray as many times as the green light blinked". The LED blinks 1-4 times at the start of the episode;
the robot must remember the count while scooping. LeRobot v2 layout (`data/`, `meta/`), images stored in the parquet files.

Extras: `openpi_assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json` = the openpi normalisation
statistics of this dataset. Per-frame sub-task labels and the episode manifest live in the code repository
(github.com/ZJU-Walker/memory_project_0922, `openpi/cluster_v5/beans/`). Restore both with `beans/ablations/00_download.sh`.
"""

BASE_CARD = """---
license: apache-2.0
tags: [openpi, pi05, robotics, yam]
---
# beans0922 pi0.5 base (knowledge insulation), step 10000

The plain pi0.5 base of the beans0922 line: openpi `pi05_base` fine-tuned on `kewalk123/yam_bean_scoop_0905_v5` with the
knowledge-insulation recipe (sub-task sentence + FAST tokens supervise the language side, flow matching trains the action
expert under a stop-gradient prefix), 10,000 updates, batch 16, lr 5e-5, EMA 0.999 (config `pi05_yam_beans0922_base`,
github.com/ZJU-Walker/memory_project_0922). `params/` is the orbax checkpoint the memory runs warm-start from
(`OPENPI_BEANS_BASE_PARAMS`). Restore with `beans/ablations/00_download.sh`.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("dataset", "base"))
    ap.add_argument("--src", default=None, help="source directory (default: the tree's own copy)")
    ap.add_argument("--workers", type=int, default=8)
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
        src = pathlib.Path(args.src or ROOT / BASE_PARAMS_REL)
        if not src.is_dir():
            sys.exit(f"base params dir missing: {src}")
        api.create_repo(BASE_REPO, repo_type="model", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=BASE_CARD.encode(), path_in_repo="README.md", repo_id=BASE_REPO, repo_type="model")
        print(f"uploading {src} -> {BASE_REPO}/params ({args.workers} workers, resumable)", flush=True)
        api.upload_large_folder(repo_id=BASE_REPO, repo_type="model", folder_path=str(src.parent), num_workers=args.workers,
                                allow_patterns=["params/**"], print_report_every=60)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
