# beans0922 ablations

## Usage on another cluster (4 x H100) -- three commands

```bash
# 1. once per machine (login node with internet; ~60 GB, 20-40 min): clone + venv + dataset + base checkpoint
curl -sO https://raw.githubusercontent.com/ZJU-Walker/memory_project_0922/main/beans/ablations/setup_other_cluster.sh
bash setup_other_cluster.sh ~/memory_project_beans0922
cd ~/memory_project_beans0922

# 2. once per row, on the 4 GPUs you own (inside your salloc / srun / job script, or a node you own): the smoke = 2 updates
GPUS=0,1,2,3 bash beans/ablations/run_vis8.sh smoke

# 3. the row itself: 3000 updates, label ramp 500, from base/10000; keep it running with nohup / tmux / your job script
GPUS=0,1,2,3 nohup bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 &
```

Rows: `run_snap.sh` (control), `run_vis8.sh` (snap + visual memory); more rows = more `run_<row>.sh`. Watch a run with
`bash beans/ablations/ablation_ctl.sh status` (or `tail -f beans/ablations/logs/train_<exp>.log`), stop it with
`ablation_ctl.sh stop`; a rerun resumes from the last checkpoint in `beans/checkpoints/<config>/<exp>/`. The launcher tries
batch 16 and falls back to 12 / 8 / 4 on OOM (4 x 80 GB usually lands at 8-12); override with `BATCH=8 BATCH_FALLBACK=4`.
Needs: git, `uv`, a CUDA-12 driver on the nodes, `wandb login` once (or `WANDB=0`). The very first data-loader start
builds the arrow cache (~40 min); every later start is fast. Nothing else needs configuring -- all paths are relative to the
clone (override with `OPENPI_BEANS_DATASET_ROOT` / `OPENPI_BEANS_BASE_PARAMS` only if you keep data elsewhere).

**Adding a row:** one config function in `openpi/src/openpi/training/beans0922_ablation_config.py` (copy `vis8_config`,
change the flag, add it to `get_configs`), one gated model flag if the row needs new code, and a two-line
`run_<row>.sh` (copy `run_vis8.sh`, change `CFG` / `EXP`). Snap's own configs must stay bit-identical.

## Background

Ablations of the LED bean-scoop memory policy ("snap": `pi05_yam_beans0922_v1`, see `../README.md`). Every row is trained
with the same 4-card recipe (`openpi/src/openpi/training/beans0922_ablation_config.py`): warm start from the beans0922
knowledge-insulation base (`beans0922_base/10000`), **3000 updates, label-write probability 1 -> 0 over the first 500**,
lr 2.5e-5, FSDP over 4 cards, largest batch that fits (launcher default 16, OOM fallback 12 / 8), checkpoints every 250
(every 500 kept). Snap's structure, window (tick 5 frames, 40 ticks), labels, sampling and losses are untouched; each
ablation is one gated flag, default off, so snap's own configs stay bit-identical.

| row | config | what differs from snap | flag |
| --- | --- | --- | --- |
| control | `pi05_yam_beans0922_ab_snap` | nothing (the recipe only) | - |
| (1) snap + visual memory | `pi05_yam_beans0922_ab_vis8` | a second fast-weight bank written every tick from the front camera and read as 8 extra input tokens next to the 8 sentence tokens | `Pi0Config.memory_vis_bank` |

## (1) snap + visual memory, in short

- **Write** (every valid tick, same rule and decay as the sentence bank): the front camera's 256 input image tokens (SigLIP +
  projector, memory-blind, stop-gradient) are pooled by 8 learned queries into 8 vectors; slot i is stored as
  key = unit(P_k v_i + e_i) (e_i a learned per-slot offset), value = unit(P_v v_i), delta rule at rate 1, decay 0.99 per tick.
  The bank is the model's `memory` reconfigured as a copy of the sentence bank's MemoryConfig (linear 512 x 2048, blank start).
- **Read** (once per tick, at the input): 8 fixed learned queries -> bank -> tanh gate (init 0.5) + RMS matched to the
  sample's own image tokens + slot embedding, appended after the 8 sentence tokens; every block sees them, the memory rows
  stay blind (attend to memory columns only). A fresh bank reads exactly zero -> masked -> tick 0 equals snap.
- **What is NOT changed**: the sentence path, the label ramp, the losses (the v3.5 side telemetry still sees snap's inert
  zero write), the data pipeline (no new image keys), serving (the server advances the visual bank once per served tick;
  `--vis-zero-read` silences its read for the reliance test).
- Telemetry (W&B `diagnostic/`): `vis_commit_count`, `vis_raw_read_rms_sum`, `vis_injected_pre_cast_rms_sum`,
  `vis_bank_norm_sum` (exact zeros for every other config).
- Tests: `openpi/src/openpi/models/pi0_v0922ab_test.py` (flag off == snap bit-for-bit; empty bank reads zero; write is
  stop-gradient and trains only `memory_vis_*`; commit / decay / invalid-tick contract; loss finite with gradients to every
  visual leaf; zeroed read cannot leak the write; sampler advances the bank in "normal" mode only) and
  `openpi/src/openpi/training/beans0922_ablation_test.py` (the configs differ in exactly the intended fields).

## Files

| file | purpose |
| --- | --- |
| `setup_other_cluster.sh` | one-shot setup on another machine: clone + `uv sync` + `00_download.sh` |
| `00_download.sh` | once per machine: dataset + norm stats + KI base checkpoint from the Hub into the tree's default paths, tokenizer caches |
| `train_ablation.sh` | generic 4-card launcher (waits for the base checkpoint and for free cards, OOM fallback ladder, resume) |
| `run_snap.sh`, `run_vis8.sh` | one row each: `[JOB=<slurm id>] [GPUS=0,1,2,3] [BATCH=16] bash beans/ablations/run_vis8.sh [smoke]` |
| `ablation_ctl.sh` | `status` / `stop` of the ablation runners and trainings |
| `hf_upload.py` (+ `_node.sh`) | the one-off pushes to the Hub (dataset done 2026-09-22 04:58; base pushed when 10k lands) |

Hub: dataset `kewalk123/yam_bean_scoop_0905_v5` (public, LeRobot layout + `openpi_assets/` norm stats), base checkpoint
`kewalk123/beans0922_pi05_base_10k` (public, `params/`). Logs of a run: `beans/ablations/logs/train_<exp>.log` and
`train_<exp>_status.log`; checkpoints `beans/checkpoints/<config>/<exp>/`; W&B project `beans0922_ablation`.

## On a new cluster

```bash
curl -sO https://raw.githubusercontent.com/ZJU-Walker/memory_project_0922/main/beans/ablations/setup_other_cluster.sh
bash setup_other_cluster.sh ~/memory_project_beans0922          # clone, venv, ~60 GB download
cd ~/memory_project_beans0922
GPUS=0,1,2,3 bash beans/ablations/run_vis8.sh smoke             # 2 updates: compile + one real batch
GPUS=0,1,2,3 nohup bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 &   # the run
```
`WORKERS`, `BATCH`, `BATCH_FALLBACK`, `MEMFRAC` (XLA memory fraction) and the `OPENPI_BEANS_*` path overrides are
environment knobs of `train_ablation.sh`; W&B needs `wandb login` once (or `WANDB=0`).
