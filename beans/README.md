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

## Throughput note (measured 2026-09-22 on iris-hgx-1)

The loader memory-maps the dataset's arrow index (the `datasets` cache) and decodes the mp4s per sample. With both on a
network filesystem the two H100s sat idle 80 % of the time (about 2 s per update at batch 16). With both on the node's
local disk the same run does 1.8 updates/s. On a shared cluster set, before launching:

```bash
export OPENPI_BEANS_DATASET_ROOT=/scr/<user>/beans0922/bean_scoop_0905_v5   # rsync -a of the dataset dir
# the arrow cache: the launcher sources cluster_v35/env.sh, which pins HF_DATASETS_CACHE to <repo>/v35/cache/huggingface/datasets
# (exporting the variable yourself is overridden), so make that in-tree directory a symlink to the local disk:
mkdir -p /scr/<user>/beans0922/hf_datasets && ln -sfn /scr/<user>/beans0922/hf_datasets <repo>/v35/cache/huggingface/datasets
```

The cache is rebuilt there on the first loader start (a few minutes from a local dataset). `beans/logs/switch_at_5000.sh` shows the
restart used here (resume from the last checkpoint). Symptom to recognise: loader workers at ~5 % CPU in `folio_wait_bit_common`
(page-fault waits on the memory-mapped arrow files) while the GPUs idle.

## Serving

`openpi/scripts/serve_yam_memory.py` serves step 2 with the same tick, write rule and read as training (the RoboMME clients
in `openpi/cluster_robomme/eval` show the request format; the YAM robot client for the LED task is
`openpi/examples/yam/client_memory_v5_led.py` from the v5 line and needs the 0920 request fields ported before the robot test).

## Ablations

`beans/ablations/` holds the ablation rows of this policy (same recipe on 4 cards, 3000 updates, label ramp 500, from the
base 10k): the control row and (1) "snap + visual memory" (`Pi0Config.memory_vis_bank`). One-time setup on another machine
= `beans/ablations/setup_other_cluster.sh` (clone, venv, dataset + base checkpoint from the Hub); one row =
`beans/ablations/run_<row>.sh`. See `beans/ablations/README.md`.

## Offline held-out probe (videos)

`beans/eval/run_heldout_videos.sh <config> <exp> <step>` walks the six development episodes (manifest split
"development": LeRobot indices 25 29 59 64 72 73, never trained on) at the training tick with the note bank carried across
ticks, decodes the subtask sentence every tick, writes it back (`self`: own sentences every tick, as deployed; `oracle`: the
label sentences) and renders the top camera with the labelled phase, the decoded sentence and the bank overlaid
(`openpi/scripts/v5_heldout_video.py`, which dispatches to the 0920 input-read prefix for these models). Output:
`beans/eval/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json}` + `status.log`. On this cluster run it inside a Slurm job you own:
`JOB=<job> GPU=<card> GRES=<cards of that job> bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v1 beans0922_v1 1500`
(one H100/H200, ~2 min per episode and mode). Elsewhere: `bash beans/eval/run_heldout_videos.sh ...` on a node with a free card.
A third pass with the bank never written: `MODES=self TAGSUF=_blank EXTRA='--intervention blank' ...` (files
`ep<idx>_self_blank.*`). Results page for checkpoint 1500 (2026-09-22): https://claude.ai/artifact/WM4QNWYaVRDAQyEiG4R6rR

Checkpoint-1500 verdict (6 dev episodes): the running blink count is read from the bank (own notes exact on 113/127 light
ticks, right count in the notes in 5/6; with the bank blank both blinks decode as "light on: 2"), but the "yellow go: scoop k
times" count ignores the notes -- own notes and label notes give the same sentence in 6/6 (right in 3/6), and the every-tick
own writes then drift 2 -> 3. Hence `pi05_yam_beans0922_v2` = v1 with change-only, confidence-gated (0.9) writes
(`V2_WRITE_RULE` in `beans0922_config.py`).

**v3 (approved 09-22 15:25) = v2 + "look before you ask" + error-driven token weight.** `pi05_yam_beans0922_v3`
(`V3_QUERY_CONTEXT`): (1) writes only on change with confidence >= 0.9 (v2); (2) `memory_v0920_query_context=True` -- each of
the 8 learned read questions is shifted, before it is asked, by its own attention over the tick's image + prompt tokens and by
the mean embedding of the last committed note, both through zero-initialised maps (fixed questions at init; the answers still
enter at the input for every block; new leaves `memory_sem_query_context_pooler/_context_proj/_prev_proj`, fresh-init by the
loader); (3) `memory_v7_hard_token_ce_weight=5.0` -- sentence tokens the model's own teacher-forced prediction gets wrong at
that tick weigh 5x (the count word is 1 token in ~50; the weight fades once learned; switches on the `v7_hard_token_count`
telemetry). Tests: `openpi/src/openpi/models/pi0_v0920_query_context_test.py`, `beans0922_test.py`. Gates at every 500 updates:
`scripts/v5_count_flip_eval.py` (true-note accuracy and flip-follow >= 0.9, blank ~1/3) + the held-out video probe (own = label
go count, right in >= 5/6). Stop rule: flip-follow < 0.5 at 2000 -> v4 = v3 + B9's slot table instead of more training.
