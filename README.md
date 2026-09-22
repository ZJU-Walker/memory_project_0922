# memory_project_0922 — LED bean-scoop memory policy: setup and ablation runs

## 1. Set up a machine (once; login node with internet; ~60 GB; needs git + [uv](https://docs.astral.sh/uv/) + a CUDA-12 driver on the GPU nodes)

```bash
curl -sO https://raw.githubusercontent.com/ZJU-Walker/memory_project_0922/main/beans/ablations/setup_other_cluster.sh
bash setup_other_cluster.sh ~/memory_project_beans0922     # clone + venv + dataset + base checkpoint + tokenizer caches
cd ~/memory_project_beans0922
wandb login                                                 # once; or WANDB=0 on every launch
```

## 2. Smoke-test a row on your 4 GPUs (2 updates: compile + one real batch)

```bash
GPUS=0,1,2,3 bash beans/ablations/run_vis8.sh smoke
```

## 3. Run a row (3000 updates, label ramp 500, warm start from the base 10k; resumes itself if relaunched)

```bash
GPUS=0,1,2,3 nohup bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 &
bash beans/ablations/ablation_ctl.sh status          # progress;  ... stop  ends it
```

| row | script | what it is |
| --- | --- | --- |
| control | `beans/ablations/run_snap.sh` | snap: 8 sentence-memory tokens |
| (1) | `beans/ablations/run_vis8.sh` | snap + visual memory: 8 sentence + 8 visual memory tokens |

Two rows on one 8-GPU node (run one smoke first, it builds the shared data cache once):

```bash
GPUS=0,1,2,3 nohup bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_vis8.out 2>&1 &
GPUS=4,5,6,7 nohup bash beans/ablations/run_snap.sh > beans/ablations/logs/run_snap.out 2>&1 &
bash beans/ablations/ablation_ctl.sh stop vis8            # stops that row only
```

Knobs (environment): `GPUS` (default `0,1,2,3`), `BATCH` (default 16, falls back to 12 / 8 / 4 on OOM), `WORKERS` (16),
`WANDB=0`. No scheduler needed: the scripts run python directly on the node you are on.
Logs: `beans/ablations/logs/train_<exp>.log`; checkpoints: `beans/checkpoints/<config>/<exp>/`; W&B project `beans0922_ablation`.

Adding a row: a config function in `openpi/src/openpi/training/beans0922_ablation_config.py` + a two-line `beans/ablations/run_<row>.sh`.
Details: `beans/ablations/README.md` (ablations), `beans/README.md` (the policy, base + memory runs, serving).
