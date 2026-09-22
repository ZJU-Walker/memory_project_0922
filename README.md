# memory_project_0922 — a self-narrated memory policy for the LED bean-scoop task, and its ablations

## What is being compared (one paragraph for the reader)

The policy is a pi0.5 VLA on the real YAM station for the task *"scoop the beans into the tray as many times as the green
light blinked"*: the green LED blinks 1–3 times at the start, then a yellow light says "go" and nothing in the scene says
the count any more. Our model,
**snap**, gives pi0.5 a small fast-weight memory that it fills with **its own words**: every tick (5 frames, 0.17 s) it
decodes a short sub-task sentence such as "scoop 2 of 3: dig and carry", writes that sentence into the memory (each word is
stored as a key/value association with the delta rule, decay 0.99 per tick), and reads the memory back through 8 fixed
learned queries as **8 extra input tokens** that every transformer block sees. The ablations ask whether a *sensory*
memory — the same kind of fast-weight bank, but filled with what the camera sees and/or where the arm is instead of a
sentence — adds anything on top of the narrated one. Every row starts from the same knowledge-insulation base checkpoint
and is trained with the same recipe (3000 updates, the first 500 with a decaying probability of writing the label sentence
instead of the model's own, lr 2.5e-5, FSDP over 4 GPUs, checkpoints kept at 1000 / 2000 / 3000); a row differs from snap in exactly one thing.

| row | script | memory tokens at the input | sensory bank written from (every tick) | update rule | status |
| --- | --- | --- | --- | --- | --- |
| control (snap) | `run_snap.sh` | 8 sentence | – | delta | runs here (Stanford 4×H200) |
| vis8 | `run_vis8.sh` | 8 sentence + 8 sensory | front camera (256 image tokens pooled into 8 slots) | delta | runs here (Stanford 4×H200) |
| vis8s | `run_vis8s.sh` | 8 sentence + 8 sensory | front camera (8 slots) + arm state (1 slot) | delta | **to run: node 1, GPUs 0–3** |
| vis8s_add | `run_vis8s_add.sh` | 8 sentence + 8 sensory | front camera (8 slots) + arm state (1 slot) | **additive** | **to run: node 1, GPUs 4–7** |
| state8 | `run_state8.sh` | 8 sentence + 8 sensory | arm state (1 slot) | delta | **to run: node 2, GPUs 0–3** |
| state8_add | `run_state8_add.sh` | 8 sentence + 8 sensory | arm state (1 slot) | **additive** | **to run: node 2, GPUs 4–7** |

*Delta rule* (W ← ρW + η(v − Wk)kᵀ) stores whether something occurred — a repeat adds nothing. *Additive rule*
(W ← ρW + η·v·kᵀ, the outer-product update of linear attention) stores how often — repeats accumulate until the decay
balances them. The sentence bank is always delta (a note must be retrieved exactly). Details: `beans/ablations/README.md`.

## 1. Set up a machine (once; needs git, [uv](https://docs.astral.sh/uv/), a CUDA-12 driver, internet, ~60 GB)

```bash
curl -sO https://raw.githubusercontent.com/ZJU-Walker/memory_project_0922/main/beans/ablations/setup_other_cluster.sh
bash setup_other_cluster.sh ~/memory_project_beans0922     # clone + venv + dataset + base checkpoint + tokenizer caches
#   LOCAL_DISK=/local/ssd bash setup_other_cluster.sh ...   # if the clone is on a network filesystem: dataset + loader cache on the node's disk (3-4x faster)
cd ~/memory_project_beans0922
wandb login                                                 # once; or WANDB=0 on every launch
```

## 2. Smoke-test once per node (2 updates: compile + one real batch; also builds the data cache, ~40 min the first time)

```bash
GPUS=0,1,2,3 bash beans/ablations/run_vis8s.sh smoke
```

## 3. The four remaining rows: two 8×H100 nodes, two rows per node (3000 updates each; a relaunch resumes)

```bash
# node 1, GPUs 0-3: vision + state bank, delta rule
GPUS=0,1,2,3 nohup bash beans/ablations/run_vis8s.sh      > beans/ablations/logs/run_vis8s.out 2>&1 &
# node 1, GPUs 4-7: vision + state bank, additive rule
GPUS=4,5,6,7 nohup bash beans/ablations/run_vis8s_add.sh  > beans/ablations/logs/run_vis8s_add.out 2>&1 &
# node 2, GPUs 0-3: state-only bank, delta rule
GPUS=0,1,2,3 nohup bash beans/ablations/run_state8.sh     > beans/ablations/logs/run_state8.out 2>&1 &
# node 2, GPUs 4-7: state-only bank, additive rule
GPUS=4,5,6,7 nohup bash beans/ablations/run_state8_add.sh > beans/ablations/logs/run_state8_add.out 2>&1 &
# any node: progress of every row / stop one row
bash beans/ablations/ablation_ctl.sh status;  bash beans/ablations/ablation_ctl.sh stop vis8s_add
```

Knobs (environment): `GPUS` (default `0,1,2,3`), `BATCH` (default 16, falls back to 12 / 8 / 4 on OOM — 4 × 80 GB usually
lands at 8–12), `WORKERS` (16 loader processes per row), `WANDB=0`. No scheduler is needed: python runs directly on the
node. Logs: `beans/ablations/logs/train_<exp>.log`; checkpoints: `beans/checkpoints/<config>/<exp>/`; W&B project
`beans0922_ablation` (the `diagnostic/vis_*` curves show the sensory bank: commits per step, bank norm, read scale).

Data and weights (public, fetched by the setup script): dataset `kewalk123/yam_bean_scoop_0905_v5` (89 episodes, LeRobot
layout), base checkpoint `kewalk123/beans0922_pi05_base_10k` (pi0.5 with knowledge insulation on this data — what every
row warm-starts from). The repo holds the **final step 10000** (its `STEP` file says so; step 5000 was published earlier,
while the base run was still going); `00_download.sh` places it under
`beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params` and the launcher warm-starts from the largest
step present. A machine that ran the download before 2026-09-22 08:30 PDT has step 5000 only: re-run
`bash beans/ablations/00_download.sh` once (a row's `train_<exp>_status.log` names the base it started from). Adding a row = one entry in `ROWS` of
`openpi/src/openpi/training/beans0922_ablation_config.py` + a two-line `beans/ablations/run_<row>.sh`.
More: `beans/ablations/README.md` (how the sensory bank works, tests), `beans/README.md` (the policy, base + memory runs, serving).
