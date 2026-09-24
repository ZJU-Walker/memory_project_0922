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
| separate SNAP-MLP3 control | `run_snap_mlp3.sh` | 5 sentence | none; different bank/gate/sampling, see section 4 |
| SNAP + visual | `run_vis8.sh` | 5 sentence + 8 auxiliary | 8 pooled front-camera vectors |
| SNAP + visual + sensory | `run_vis8s.sh` | 5 sentence + 8 auxiliary | 8 camera vectors + 1 joint-state vector |
| SNAP + sensory | `run_state8.sh` | 5 sentence + 8 auxiliary | 1 joint-state vector |
| optional additive controls | `run_vis8s_add.sh`, `run_state8_add.sh` | same as corresponding row | same sources, additive update |

Here “sensory” means **proprioceptive joint state**. Visual + sensory share one auxiliary bank, not two independent banks.
The four main rows all use the delta rule. It corrects the residual of a retrieved association; it is not a guarantee of
perfect recall or exact counting. The optional additive rows accumulate associations and are secondary controls.

The four main rows and additive controls use the same KI base **10000**, split (77 train / 6 val / 6 test), seed 42, 4 GPUs,
learning-rate schedule, v4e onset sampling/loss weights, and decay 0.999/tick. By user choice, global batch is **16 on H200**
and **12 on H100**, both without accumulation. This is a training-budget difference: at equal updates H200 sees 4/3 as many
windows, so cross-cluster comparisons are not strictly bank-only ablations. The launcher never silently reduces batch.
Every row trains its own A (250 updates, label writes), then loads **all its own A parameters** into B (3000 updates,
predicted-content writes, fresh optimizer; no 500-step label-write ramp).
B's in-window writer is argmax **under teacher forcing**, not free-running autoregressive training. Both phases retain
label-based past-sentence prefill; serving/evaluation decode autoregressively. Keep this distinction in experiment reports.

The main recipe's write gate keeps minimum word probability 0.3 and change-only writes; debounce is **one tick** so a one-tick light
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
GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 bash beans/ablations/run_vis8s.sh smoke
```

Smoke executes **A2 → B2**, including checkpoint loading. It has separate experiment names and no W&B.
Run one main row per four available GPUs (choose the appropriate line on each node; not all on the same cards):

```bash
mkdir -p beans/ablations/logs
# Stanford 4 H200: SNAP; when outside an existing allocation set JOB=<allocation> GRES=4.
setsid nohup env GPUS=0,1,2,3 BATCH=16 WORKERS=8 A_STEPS=250 STEPS=3000 bash beans/ablations/run_snap.sh > beans/ablations/logs/run_slot_snap.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + visual
setsid nohup env GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 A_STEPS=250 STEPS=3000 bash beans/ablations/run_vis8.sh > beans/ablations/logs/run_slot_vis8.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + visual + sensory
setsid nohup env GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 A_STEPS=250 STEPS=3000 bash beans/ablations/run_vis8s.sh > beans/ablations/logs/run_slot_vis8s.out 2>&1 < /dev/null &
# Other node / four free cards: SNAP + sensory
setsid nohup env GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 A_STEPS=250 STEPS=3000 bash beans/ablations/run_state8.sh > beans/ablations/logs/run_slot_state8.out 2>&1 < /dev/null &
```

An 8-GPU node can run two rows with `GPUS=0,1,2,3` and `GPUS=4,5,6,7`.
4×H200 uses `BATCH=16 ACCUM=1` (4 samples/GPU). 4×H100 uses `BATCH=12 ACCUM=1` (3 samples/GPU).
There is no gradient accumulation in either requested recipe. No automatic OOM fallback changes the batch.
A250/B3000 and the learning rate are unchanged; record the hardware/batch difference in reports and smoke-test each target node.
Each script chains A→B without user intervention. Rerunning the same command resumes a completed numeric checkpoint;
a completed A is skipped. To start a new experiment use `RUN_NAME=slot_vis8_seed42_retry`, never delete an active directory.
If you already launched the earlier H100 batch-8 recipe, stop that row and use a new name such as
`RUN_NAME=slot_vis8s_b12`; do not resume its batch-8 checkpoint as a batch-12 ablation. The recipe guard enforces this.

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
H100 commands use `WORKERS=4` conservatively because past-image replay adds host-memory pressure. Check the job's RAM limit
before increasing workers; GPU count alone does not determine safe loader parallelism.
Defaults: `ACCUM=1`, `WORKERS=8`, `A_STEPS=250`, `STEPS=3000`, `BATCH=16`, `GPUS=0,1,2,3`.
`JOB` enables an overlapping Slurm step; omit it when already on allocated GPUs. 1 GB keep-alives are permitted by
the launcher and must be left running. Source/data details: [mechanism](beans/ablations/README.md), [run history](beans/README.md).

## 4. SNAP-MLP3: separate sentence-only control

Row **`snap_mlp3`**, recipe **`template_slot_snap_mlp3_v1`**. This is not a full B9 reproduction and not a bank-only
comparison with the four main rows. Only these controls change from current SNAP:

| setting | `snap` | `snap_mlp3` |
| --- | --- | --- |
| sentence-bank hidden layers | none (linear) | **1024 / 1024 / 1024** |
| inner memory update | delta on final matrix | delta on final **1024 x 2048** matrix only |
| predicted-sentence write confidence | minimum valid-token probability >= 0.3 | **mean valid-token probability >= 0.9** |
| episode-start / ordinary / transition-window probability | 15% / 15% / 70% | **25% / 25% / 50%** |
| transition start range | 0-25 frames before sentence change | **0-50 frames** before sentence change |

The hidden layers still learn through the outer training optimizer; per-tick delta writes and decay do not update them.
Unchanged: five direct template reads at the **input**, whole-sentence writes, occupancy masks, no visual/sensory bank,
change-only writes with one-tick debounce, no retraction, decay **0.999**, no state masking, RTC **15**, stride **5 frames**,
window **40 ticks**, TBPTT **25**, original labels/split/seed, KI base **10000**, and own **A250 -> B3000**.
A uses oracle label writes; the mean-0.9 gate applies to B's predicted writes and deployment, not A's label writes.
Both phases retain the current lr **2.5e-5**, warmup 100 and v4e loss settings (onset 6, wrong sentence token 5,
decision-tick CE 0.2). Loss weights have **not** been reverted to B9. H100 uses **BATCH=12 ACCUM=1**.

### Switch a four-H100 node from vis8 to SNAP-MLP3

Execute on the **compute node running vis8**, inside the same checkout. Adjust only the `cd` path for another cluster.
Stop before pulling; `stop vis8` targets that row in this checkout, not vis8s, other projects, keep-alives, or the allocation.
Existing vis8 checkpoints/logs are retained. No download, deletion or directory rename is needed because the new row has
its own namespace and always starts its A from KI base/10000, never vis8 or linear SNAP.

```bash
cd /fs/scratch/PAS2099/memory_project_beans0922
bash beans/ablations/ablation_ctl.sh status vis8
bash beans/ablations/ablation_ctl.sh stop vis8
bash beans/ablations/ablation_ctl.sh status vis8  # should report 0 matching processes
git pull --ff-only origin main
nvidia-smi
```

For a newer long W&B API key, use an exported environment variable in this launch shell (the pinned wandb 0.19.11 CLI's
`login --relogin` has a legacy 40-character check). Do not put the real key in a script or Git:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
```

The detached chain smoke-tests **A2 -> B2** first (separate checkpoint names, W&B off), then starts **A250 -> B3000** only if
the smoke succeeds. The nonlinear bank has not been GPU-tested on every target cluster; an OOM stops the chain instead of
silently lowering batch. Run only one row on these four cards.

```bash
mkdir -p beans/ablations/logs
setsid nohup env \
  GPUS=0,1,2,3 BATCH=12 ACCUM=1 WORKERS=4 \
  A_STEPS=250 STEPS=3000 RUN_NAME=slot_snap_mlp3_b12 \
  bash -c '
    set -euo pipefail
    bash beans/ablations/run_snap_mlp3.sh smoke
    exec bash beans/ablations/run_snap_mlp3.sh
  ' > beans/ablations/logs/run_slot_snap_mlp3_b12.out 2>&1 < /dev/null &

tail -f beans/ablations/logs/run_slot_snap_mlp3_b12.out
# Detailed logs (Ctrl+C exits only tail):
# smoke: train_smoke_slot_snap_mlp3_b12_A.log, then ..._B.log
# main:  train_slot_snap_mlp3_b12_A.log, then ..._B.log
```

W&B project stays `beans0922_ablation`; main run names are **`slot_snap_mlp3_b12_A`** and **`slot_snap_mlp3_b12_B`**.
Checkpoints: `beans/checkpoints/pi05_yam_beans0922_ab_snap_mlp3_A/slot_snap_mlp3_b12_A/250/params` and
`beans/checkpoints/pi05_yam_beans0922_ab_snap_mlp3/slot_snap_mlp3_b12_B/{1000,2000,3000}/params`.
`nohup` does not extend a Slurm allocation or keep an interactive allocation/step alive after its owning shell exits;
keep the allocation active, or use the cluster's batch-job submission workflow.

### Sentence similarity and slot interference

The CPU-only probe measures the real checkpoint's sentence-encoding, raw-value, whitened-value, template-key and
**post-MLP hidden-feature** cosine matrices. It also tests single-write recall and sequential overwriting (latest value
per template, both reference and reverse order). Same-template keys are intentionally identical; high similarity there
is **not** a failure. Count distinctions should survive in the values and readback. Different-template hidden-feature
similarity matters for delta-write interference even when input template keys are separated.

Before training, measure the exact seed-42 stage-A initialization (same KI graft and frozen-parameter cast as training):

```bash
bash beans/ablations/measure_sentence_geometry.sh \
  --config-name pi05_yam_beans0922_ab_snap_mlp3_A --initialize \
  --output-dir beans/ablations/diagnostics/snap_mlp3_init_seed42
```

After checkpoint B1000 exists (change the step to 2000/3000 for subsequent measurements):

```bash
bash beans/ablations/measure_sentence_geometry.sh \
  --config-name pi05_yam_beans0922_ab_snap_mlp3 \
  --params beans/checkpoints/pi05_yam_beans0922_ab_snap_mlp3/slot_snap_mlp3_b12_B/1000/params \
  --output-dir beans/ablations/diagnostics/slot_snap_mlp3_b12_B1000
```

Reports are create-only `sentence_geometry.json` files including the full matrices, sentence/template mapping and
parameter hash. This is oracle-association geometry, **not** task accuracy or a test of the 0.9 prediction gate;
initialization measurements are explicitly labelled untrained. The helper uses no GPU and does not stop/reconfigure training.

Initial CPU measurement (2026-09-23, KI10000 + fresh SNAP-MLP3, seed 42; **not a trained checkpoint**):

| sentence pair | encoding cosine before value whitening | actual whitened write-value cosine |
| --- | --- | --- |
| yellow-go scoop 2 vs 3 times | 0.958462 | 0.106708 |
| yellow-go scoop 1 vs 2 times | 0.927738 | 0.013050 |
| scoop 1 of 2 vs 1 of 3, dump-and-return | 0.943655 | 0.163590 |

Across all 20 sentences the largest off-diagonal write-value cosine is **0.233997**. Same-template keys have cosine 1
by design. Different-template post-MLP hidden-feature cosines range **-0.549037 to -0.001690**: no same-direction collapse,
but they are not orthogonal, so interference remains. All 20 isolated writes recall at cosine approximately 1.
After writing all sentences in reference/reverse order, latest-value identification is **10/10** slot/order cases
(**6/6** cases with multiple candidate sentences; four singleton cases are trivial). Read-to-target cosine falls as low as
**0.519423**, so correct nearest-value identification must not be interpreted as perfect retention or policy success.
Model-parameter SHA256: `7ab9b8e2184a6b3233c8fe9418aaedae1b2a07ae4fc0f02abe913cd99119371c`.
Repeat on A250 and B checkpoints to check whether training improves or collapses these distinctions.
