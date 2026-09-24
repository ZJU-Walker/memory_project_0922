# memory_project_0922 — template-slot SNAP and ablations

## What is being compared

Current recipe: **template_slot_ab_v1 (2026-09-23), per-row A250 → B3000**.
SNAP writes a whole narrated sentence into a fast-weight sentence bank. It retains the current **input-layer injection**
interface, but disables token-level writes, token pointer and 48-token read-back. Every tick it reads all induced template
addresses in parallel, then lets the transformer use those retrieved vectors. This is **not an exact B9 reproduction**:
B9's template writer is paired with direct template reads instead of learned retrieval queries.

Templates are induced from the supplied training sentence vocabulary by token differences; no beans phase names or fixed
five-way task rules are coded into discovery. The current 20-sentence vocabulary produces **5 addresses**. This is not
open-vocabulary automatic task discovery: a new task still needs its training sentence vocabulary/labels; inspect the induced
groups, and out-of-vocabulary sentences are rejected as writes. A slot is a sentence-template family, not one slot per word.
Unwritten addresses are masked even if other writes cause fast-weight cross-talk.

| row | script | input memory tokens | auxiliary bank writes |
| --- | --- | --- | --- |
| SNAP | `run_snap.sh` | 5 sentence | none |
| SNAP + visual | `run_vis8.sh` | 5 sentence + 8 auxiliary | 8 pooled front-camera vectors |
| SNAP + visual + sensory | `run_vis8s.sh` | 5 sentence + 8 auxiliary | 8 camera vectors + 1 joint-state vector |
| SNAP + sensory | `run_state8.sh` | 5 sentence + 8 auxiliary | 1 joint-state vector |
| optional additive controls | `run_vis8s_add.sh`, `run_state8_add.sh` | same as corresponding row | same sources, additive update |

Here “sensory” means **proprioceptive joint state**. Visual + sensory share one auxiliary bank, not two independent banks.
The four main rows all use the delta rule. It corrects the residual of a retrieved association; it is not a guarantee of
perfect recall or exact counting. The optional additive rows accumulate associations and are secondary controls.

All rows use the same KI base **10000**, split (77 train / 6 val / 6 test), seed 42, 4 GPUs,
learning-rate schedule, v4e onset sampling/loss weights, and decay 0.999/tick. By user choice, global batch is **16 on H200**
and **8 on H100**, both without accumulation. This is a training-budget difference: at equal updates H200 sees twice as many
windows, so cross-cluster comparisons are not strictly bank-only ablations. The launcher never silently reduces batch.
Every row trains its own A (250 updates, label writes), then loads **all its own A parameters** into B (3000 updates,
predicted-content writes, fresh optimizer; no 500-step label-write ramp).
B's in-window writer is argmax **under teacher forcing**, not free-running autoregressive training. Both phases retain
label-based past-sentence prefill; serving/evaluation decode autoregressively. Keep this distinction in experiment reports.

The write gate keeps minimum word probability 0.3 and change-only writes; debounce is **one tick** so a one-tick light
observation is not discarded. Tick = 5 frames, window = 40 ticks, TBPTT = 25, no state masking.
Auxiliary banks replay **past-only real observations** before the sampled window (stop-gradient); full-episode rollout carries
their state between ticks. Capacity 320 past ticks covers this dataset; longer tasks fail explicitly and need a larger bound.
Visual replay adds decoding/encoder cost; do not silently turn it off for one row.

## 1. New machine setup

Needs git, uv, a CUDA-12 driver, enough CPU RAM for workers/checkpoint saves, and dataset/checkpoint storage.
The existing setup/download script format is unchanged:

```bash
curl -sO https://raw.githubusercontent.com/ZJU-Walker/memory_project_0922/main/beans/ablations/setup_other_cluster.sh
bash setup_other_cluster.sh ~/memory_project_beans0922
cd ~/memory_project_beans0922
# Optional node-local data/cache: LOCAL_DISK=/your/local/ssd on the setup/download command.
# W&B: openpi/.venv/bin/wandb login; or export WANDB=0
```

## 2. Existing cluster: stop old training, update, clean obsolete active directories

Run inside the checkout on the node running that checkout's old training. Fetching does not change live source files.
The stop helper is taken from the new commit before pulling; it matches this UID and checkout only, includes child workers,
and never cancels the Slurm allocation or kills `cluster_scripts/train_hs.py` keep-alives.

```bash
cd ~/memory_project_beans0922
git fetch origin main
ctl=$(mktemp /tmp/beans0922-ctl.XXXXXX.py)
git show origin/main:beans/ablations/manage_runs.py > "$ctl"
python3 "$ctl" status --root "$PWD"
python3 "$ctl" stop --root "$PWD"
git pull --ff-only origin main
bash beans/ablations/ablation_ctl.sh archive             # preview
bash beans/ablations/ablation_ctl.sh archive --apply     # recoverable cleanup
bash beans/ablations/00_download.sh                     # ensures KI base/10000 + data
```

On the Stanford development checkout the corresponding remote is `origin0922`, not `origin`.
Archive moves only old `ab_<row>`, `smoke_ab_<row>`, and `probe_<row>` experiment directories into
`beans/checkpoints/_archive/<timestamp>/`. It **does not free disk space**; no permanent deletion of checkpoints,
datasets, evaluation results or historical v4e runs is performed. New runs use `slot_<row>_A/B` and never resume old token-bank runs.
Do not use broad `pkill python` or `scancel <allocation>`: those can kill keep-alives or unrelated work.

## 3. Test, then train (same launch format on each cluster)

```bash
bash beans/ablations/run_tests.sh cpu
GPUS=0,1,2,3 BATCH=16 WORKERS=8 bash beans/ablations/run_snap.sh smoke
# For an auxiliary row, smoke that row on its target hardware too:
GPUS=0,1,2,3 BATCH=8 ACCUM=1 WORKERS=8 bash beans/ablations/run_vis8s.sh smoke
```

Smoke executes **A2 → B2**, including checkpoint loading. It has separate experiment names and no W&B.
Run one main row per four available GPUs (choose the appropriate line on each node; not all on the same cards):

```bash
mkdir -p beans/ablations/logs
# Stanford 4 H200: SNAP; when outside an existing allocation set JOB=<allocation> GRES=4.
setsid nohup env GPUS=0,1,2,3 BATCH=16 WORKERS=8 A_STEPS=250 STEPS=3000 bash beans/ablations/run_snap.sh > beans/ablations/logs/run_slot_snap.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + visual
setsid nohup env GPUS=0,1,2,3 BATCH=8 ACCUM=1 WORKERS=8 A_STEPS=250 STEPS=3000 bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_slot_vis8.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + visual + sensory
setsid nohup env GPUS=0,1,2,3 BATCH=8 ACCUM=1 WORKERS=8 A_STEPS=250 STEPS=3000 bash beans/ablations/run_vis8s.sh > beans/ablations/logs/run_slot_vis8s.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + sensory
setsid nohup env GPUS=0,1,2,3 BATCH=8 ACCUM=1 WORKERS=8 A_STEPS=250 STEPS=3000 bash beans/ablations/run_state8.sh > beans/ablations/logs/run_slot_state8.out 2>&1 < /dev/null &
```

An 8-GPU node can run two rows with `GPUS=0,1,2,3` and `GPUS=4,5,6,7`.
4×H200 uses `BATCH=16 ACCUM=1` (4 samples/GPU). 4×H100 uses `BATCH=8 ACCUM=1` (2 samples/GPU).
There is no gradient accumulation in either requested recipe. No automatic OOM fallback changes the batch.
A250/B3000 and the learning rate are unchanged; record the hardware/batch difference in reports and smoke-test each target node.
Each script chains A→B without user intervention. Rerunning the same command resumes a completed numeric checkpoint;
a completed A is skipped. To start a new experiment use `RUN_NAME=slot_vis8_seed42_retry`, never delete an active directory.

```bash
bash beans/ablations/ablation_ctl.sh status
bash beans/ablations/ablation_ctl.sh stop vis8s           # this checkout, this row only
tail -f beans/ablations/logs/train_slot_snap_A.log        # then train_slot_snap_B.log
```

Checkpoint locations:

- A: `beans/checkpoints/pi05_yam_beans0922_ab_<row>_A/slot_<row>_A/250/params`
- B: `beans/checkpoints/pi05_yam_beans0922_ab_<row>/slot_<row>_B/{1000,2000,3000}/params`

Step names retain this repository's existing zero-based convention (including Step 0): A250/B3000 name the final
checkpoint steps. Smoke A2/B2 likewise ends at checkpoint step 2.

W&B project: `beans0922_ablation`; standard sentence/flow losses and sensory-bank telemetry retain their existing names.
Defaults: `ACCUM=1`, `WORKERS=8`, `A_STEPS=250`, `STEPS=3000`, `BATCH=16`, `GPUS=0,1,2,3`.
`JOB` enables an overlapping Slurm step; omit it when already on allocated GPUs. 1 GB keep-alives are permitted by
the launcher and must be left running. Source/data details: [mechanism](beans/ablations/README.md), [run history](beans/README.md).
