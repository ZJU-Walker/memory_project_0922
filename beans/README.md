# beans0922 — LED bean-scoop memory policy (real YAM station)

The real-robot line of the 0920 memory structure: a pi0.5 that writes its own sub-task sentence into a small fast-weight bank
every tick and reads it back through 8 learned queries at the input (see `openpi/src/openpi/training/robomme_0920_config.py`
for the structure and `beans0922_config.py` for this task). Task: "scoop the beans into the tray as many times as the green
light blinked" — the count must be remembered from the LED blinks at the start of the episode.

## What runs

| step | config | from | length | cards |
| --- | --- | --- | --- | --- |
| 1 | `pi05_yam_beans0922_base` | public `pi05_base` | 10k updates, batch 16 | 2 GPUs, FSDP 2 |
| 2 | `pi05_yam_beans0922_v1` | step 1's `beans0922_base/10000` | 5k updates, batch 4, label-write ramp 500 | 2 GPUs, FSDP 2 |

Step 1 is the plain knowledge-insulation base (sub-task sentence + FAST tokens supervise the language side, flow matching
trains the action expert under a stop-gradient prefix). Step 2 adds the memory: tick 5 frames (0.17 s, so every LED blink is
seen), 40-tick windows, TBPTT 25, own writes every tick, label content with probability 1 -> 0 over the first 500 updates (fully self-written from 500 on), no state masking.

## Setup on a new machine

```bash
git clone <this repo> memory_project_beans0922 && cd memory_project_beans0922/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen            # python 3.11 venv at openpi/.venv
```

Data and weights (all paths are relative to the repo root; override any of them with the environment variables in the table):

| what | default location | override |
| --- | --- | --- |
| LeRobot dataset `yam/bean_scoop_0905_v5` (89 episodes, 71,089 frames, 53 GB) | `v5/data/lerobot/yam/bean_scoop_0905_v5` | `OPENPI_BEANS_DATASET_ROOT` |
| norm stats | `v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json` | `OPENPI_BEANS_ASSETS_DIR` (dir above `yam/`) |
| pi05 base weights for step 1 | `gs://openpi-assets/checkpoints/pi05_base/params` (downloaded on first use) | `OPENPI_BEANS_PI05_BASE` |
| step-1 checkpoint for step 2 | `beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params` | `OPENPI_BEANS_BASE_PARAMS` |
| labels + manifest (in the repo) | `openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json`, `beans_episode_manifest_0905_v1.json` | — |

To rebuild the dataset from the raw demos (`data/0905beans_{1,2,3}`, 89 demo folders with the three camera mp4s, joint
streams and the LED/go signals): `HF_LEROBOT_HOME=<root>/v5/data/lerobot python examples/yam/convert_yam_data_to_lerobot.py
--episode-manifest data/0905beans_episode_manifest_v1.json --repo-name yam/bean_scoop_0905_v5` (about 25 min on a CPU node).
`MEMORY_PROJECT_ROOT` overrides the repo root if the tree is not where the scripts live.

## Launch

```bash
# this cluster (Slurm, inside an allocation): the whole chain, detached
JOB=<job id> GPUS=0,1 bash beans/logs/beans0922_ctl.sh start
# another machine where you already own the GPUs: no JOB -> python runs directly
GPUS=0,1 bash beans/logs/beans0922_ctl.sh start
bash beans/logs/beans0922_ctl.sh status | stop     # smoke_base / smoke_mem / base / mem run one stage
```

The launcher refuses to start while another process holds more than 2 GB on the chosen cards, resumes an experiment dir
that already holds a numeric checkpoint, and in step 2 retries once at batch 2 if batch 4 runs out of memory. Logs:
`beans/logs/train_<exp>.log`, status lines in `beans/logs/train_beans0922_status.log`. W&B project `beans0922`.

## Serving

`openpi/scripts/serve_yam_memory.py` serves step 2 with the same tick, write rule and read as training (the RoboMME clients
in `openpi/cluster_robomme/eval` show the request format; the YAM robot client for the LED task is
`openpi/examples/yam/client_memory_v5_led.py` from the v5 line and needs the 0920 request fields ported before the robot test).

## Ablations

`beans/ablations/` holds the ablation rows of this policy (same recipe on 4 cards, 3000 updates, label ramp 500, from the
base 10k): the control row and (1) "snap + visual memory" (`Pi0Config.memory_vis_bank`). One-time setup on another machine
= `beans/ablations/setup_other_cluster.sh` (clone, venv, dataset + base checkpoint from the Hub); one row =
`beans/ablations/run_<row>.sh`. See `beans/ablations/README.md`.
