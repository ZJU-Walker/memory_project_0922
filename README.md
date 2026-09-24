# memory_project_0922 — template-slot SNAP and ablations

## What is being compared

Current recipe: **template_slot_ab_v1 (2026-09-23), per-row A250 → B3000**.
SNAP writes a whole narrated sentence into a fast-weight sentence bank. It retains the current **input-layer injection**
interface, but disables token-level writes, token pointer and 48-token read-back. Every tick it reads all induced template
addresses in parallel, then lets the transformer use those retrieved vectors. This is **not an exact B9 reproduction**:
B9's template writer is paired with direct template reads instead of learned retrieval queries.

Separate A9-aligned rows retain **eight conditioned queries at layer 8** instead: the sentence-slot
`snap_mlp3_a9align` ([section 5](#5-a9b9-aligned-snap-mlp3)) and its token-writer control
`snap_token_mlp3_a9align` ([section 6](#6-token-mlp3-a9-aligned-writer-control-new-node)).

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

## 5. A9/B9-aligned SNAP-MLP3

Independent row **`snap_mlp3_a9align`**, recipe **`a9align_snap_mlp3_v1`** (2026-09-24).
Do not resume an existing input-read `snap` or `snap_mlp3` checkpoint into this row.
The previous rows and their launch commands are unchanged.

The model is cloned directly from old **A9 for A / B9 for B**, with **RTC 6 -> 15** and the B-only execution optimization below:

- Three 1024-wide hidden layers; whole-sentence template keys and whitened values; inner delta updates only the output
  matrix (delta rate 1). Hidden-layer parameters remain trainable by the outer optimizer.
- **Eight conditioned read queries, layer-8 injection**. Instruction-context standardization and previous-sentence
  conditioning restored. No direct five-address input read, token writes, pointer bonus, or token read-back.
- Original layer-8 injection scaling/gates and memory-blind attention. No visual/sensory content injection; the frozen,
  inactive legacy visual subsystem/columns are retained exactly as in A9/B9 rather than silently changing their structure.
- Decay multiplier **0.99/tick**, training state-mask probability **0.5**, sentence onset / hard-token / post-motion
  decision weights **1 / 1 / 1**. No v4e extra weighting.
- Original 0905 labels and **77 train / 6 development / 6 final-test** split; stride **5**, max **40** ticks,
  buckets **14/27/40**, TBPTT **25**, history prefill **16**, action horizon **50**, lookahead **0**.
- Window mixture episode-start / ordinary-slice / transition = **25/25/50**, starts **0-75 frames** before a transition.
  This intentionally restores old A9's pad75 instead of the earlier input-read MLP3's pad50.
- A oracle-label writes; B teacher-forced argmax content writes with **mean valid-token probability >=0.9**, change-only,
  retry-until-committed. Both retain label-history prefill; B is not fully autoregressive rollout training.
- A LR **5e-5**, B LR **2.5e-5**, each warmup **100**; old AdamW/freeze filters/clipping, no EMA, seed **42**.
- **A keeps image encoding inside the tick scan; B sets `memory_v0920_vision_outside_scan=True`** (2026-09-24).
  B batches the frozen image tower before the scan to avoid tick-level recomputation. This does not move memory injection
  back to the input or change the checkpoint parameter layout. Different batching need not be bit-identical.
  The running A is not restarted or modified; the existing chain launches B in a fresh Python process and reads this config.

Intentional non-model differences: old A9's `stageB6a/keep_499` initialization is unavailable. This row loads the
current **KI base/10000** non-memory weights and initializes memory afresh; it does **not** reuse old linear A250.
B strictly loads **its own A500**, including every memory parameter, with a fresh optimizer. Default lengths are
**A500 -> B3000**, retaining this repo's Step-0-inclusive checkpoint-label convention. H200 global batch **16**,
H100 **12**, accumulation **1**, FSDP across all four cards. Historical A9/B9 actually used global batch **8**, so this
is a recipe-aligned new experiment, not a controlled reproduction. A keeps 250/500; B saves every 250 and keeps
1000/2000/3000 plus the two most recent rolling checkpoints.

Run on the allocated GPU node (stop only a specifically selected old row if needed; never cancel the allocation or
the separate 1GB keepalive). On another cluster complete the setup/download instructions above first.

```bash
bash beans/ablations/ablation_ctl.sh status
# Only when replacing an old SNAP run:
# bash beans/ablations/ablation_ctl.sh stop snap
mkdir -p beans/ablations/logs
setsid nohup env GPUS=0,1,2,3 BATCH=16 ACCUM=1 WORKERS=8 \
  A_STEPS=500 STEPS=3000 RUN_NAME=slot_snap_mlp3_a9align_b16 \
  bash -c 'set -euo pipefail
    bash beans/ablations/run_snap_mlp3_a9align.sh smoke
    exec bash beans/ablations/run_snap_mlp3_a9align.sh
  ' > beans/ablations/logs/run_slot_snap_mlp3_a9align_b16.out 2>&1 < /dev/null &

tail -f beans/ablations/logs/run_slot_snap_mlp3_a9align_b16.out
# Formal-stage log after A2 -> B2 smokes pass:
# tail -f beans/ablations/logs/train_slot_snap_mlp3_a9align_b16_A.log
```

For a four-H100 node set `BATCH=12 WORKERS=4 RUN_NAME=slot_snap_mlp3_a9align_b12` and use that name in the log paths.
For launch from outside a Slurm allocation add `JOB=<your_job_id> GRES=4 CPUS=24` to the environment above.
Smokes are isolated, W&B off; the formal chain uses project `beans0922_ablation` and separate `_A` / `_B` run names.
Detached launch survives terminal/Codex closure while the Slurm allocation remains active. No automatic batch reduction.
Stop this new row only: `bash beans/ablations/ablation_ctl.sh stop snap_mlp3_a9align`.

## 6. Token-MLP3: A9-aligned writer control (new node)

Row **`snap_token_mlp3_a9align`**, recipe **`a9align_token_mlp3_v1`**. This is a fresh, separate experiment,
not a resume of slot SNAP or historical v4e. User approved 2026-09-24: run on a **new node; do not stop the old training**.

Relative to `snap_mlp3_a9align`, exactly four model config fields change:

```python
memory_v5_slot_keys = False
memory_v5_whiten_values = False
memory_v6_token_writes = True
memory_v6_whiten_keys = True
```

The entire changed sentence still passes one gate; then each valid token is written sequentially with its contextual key
and standardized/projected token-embedding value (`memory_v6_value_standardize=True`, unchanged). Padding is not written.
One decay per **tick**, not per token. The existing first-token convention is unchanged: positions 0 and 1 both use the
first causal state, so their address can coincide; do not interpret token writes as disjoint allocated array slots.
Context whitening happens **before** the nonlinear MLP and does not guarantee post-MLP orthogonality or perfect recall.

Everything else follows section 5: **3x1024** hidden layers, output-matrix-only delta at rate 1, **8 conditioned queries at
layer 8**, decay 0.99, state mask 0.5, RTC15, stride5, T40, TBPTT25, prefill16, sampling25/25/50 with pad75, original
A9/B9 losses/LRs/freeze rules, seed42 and the same dataset/split. **No pointer, input-layer read, 48-token read-back,
visual or sensory content injection.** Sentence buffers remain 48 tokens, which is not a 48-token read-back mechanism.
A keeps inside-scan vision; B uses the already-approved outside-scan batching.

A500 uses true-label token writes and starts from KI base/10000 non-memory weights plus fresh memory. B3000 strictly
loads **its own token-A500**, including the token projections, then uses predicted-content writes (teacher-forced argmax,
mean valid-token probability >=0.9, change-only, retry-until-committed). Both stages retain label-history prefill, now
through the same token writer. Save every250; A retains250/500; B permanently retains1000/2000/3000 plus two latest saves.
No old slot checkpoint is imported. Token key/value projections and per-sentence association counts differ: this is a
writer-representation ablation, not equal parameter count or equal compute.

### Launch on the new four-H100 compute node

Use the **new node's allocation**, not the old slot training node or a login node. This assumes the existing shared checkout,
environment and downloaded data/base; a brand-new cluster must first follow section 1. No kill, archive, rename or download
is needed for an already-set-up cluster. Keep any existing 1GB keep-alives; the launcher permits them and never cancels jobs.

```bash
cd /fs/scratch/PAS2099/memory_project_beans0922
git pull --ff-only origin main

(
  set -euo pipefail
  test -x openpi/.venv/bin/python
  test -d beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params
  nvidia-smi
  mkdir -p beans/ablations/logs
  setsid nohup env \
    JOB="${SLURM_JOB_ID:-}" GRES=4 CPUS=24 GPUS=0,1,2,3 \
    BATCH=12 ACCUM=1 WORKERS=4 A_STEPS=500 STEPS=3000 \
    RUN_NAME=token_mlp3_a9align_b12 \
    OPENPI_BEANS_BASE_PARAMS="$PWD/beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params" \
    bash -c '
      set -euo pipefail
      unset OPENPI_BEANS_AB_A_PARAMS OPENPI_BEANS_AB_CHECKPOINT_ROOT
      bash beans/ablations/run_snap_token_mlp3_a9align.sh smoke
      exec bash beans/ablations/run_snap_token_mlp3_a9align.sh
    ' >> beans/ablations/logs/run_token_mlp3_a9align_b12.out 2>&1 < /dev/null &
  echo "Detached launcher PID: $!"
)
tail -f beans/ablations/logs/run_token_mlp3_a9align_b12.out
```

The chain first does **A2 -> B2** real-data smoke training/loading (separate smoke checkpoints), then **fresh A500 -> B3000**.
It stops on smoke failure rather than launching the main run. First JAX compilation can take several minutes, especially
with the token loop; the CPU regressions are not a substitute for this device-specific smoke. Global H100 batch12 differs
from H200 slot batch16: equal steps expose H200 to 4/3 as many windows. A strict comparison needs matched batch/budget.
The detached launcher survives terminal/app disconnect **while the GPU allocation remains alive**; allocation expiry or
cancellation still stops training. Do not end the allocation or its owning interactive shell.

Actual model progress is in these files, created as each stage starts (the outer log only reports stage launches):

```bash
# During the first smoke A; switch to the corresponding B file when the chain advances.
tail -f beans/ablations/logs/train_smoke_token_mlp3_a9align_b12_A.log
# Formal A, then formal B:
tail -f beans/ablations/logs/train_token_mlp3_a9align_b12_A.log
tail -f beans/ablations/logs/train_token_mlp3_a9align_b12_B.log
```

W&B stays `beans0922_ablation`, with runs `token_mlp3_a9align_b12_A/B`. Checkpoints:

- A: `beans/checkpoints/pi05_yam_beans0922_ab_snap_token_mlp3_a9align_A/token_mlp3_a9align_b12_A/{250,500}/params`
- B: `beans/checkpoints/pi05_yam_beans0922_ab_snap_token_mlp3_a9align/token_mlp3_a9align_b12_B/{1000,2000,3000}/params`

### CPU-only token geometry

The existing diagnostic also supports this token writer. It measures isolated association recall, whole-sentence writes
(including within-sentence interference), pre/post-MLP context cosines, and latest-token recall after writing all reference
sentences forward/reversed. Repeated contexts are scored against their latest value, not every historical value. This uses
exact write keys, **not** the eight learned reader queries or policy success. `--initialize` is explicitly untrained.

```bash
bash beans/ablations/measure_sentence_geometry.sh \
  --config-name pi05_yam_beans0922_ab_snap_token_mlp3_a9align_A --initialize \
  --output-dir beans/ablations/diagnostics/token_mlp3_a9align_init_seed42
# Later: replace --initialize with --params /absolute/path/to/token/checkpoint/params
# and choose a new output directory. Reports are create-only and use CPU, not the training GPUs.
```

Optional storage override: add `OPENPI_BEANS_AB_CHECKPOINT_ROOT=/absolute/checkpoint/root` to the launch environment.
The A/B saves, resume lookup and B's A-checkpoint loader all use this root; KI source weights and logs remain in the
repository. The recipe guard records the override and refuses to silently resume from a different root. Node-local
`/scr` is **not permanent storage**: archive important checkpoints before the node/allocation is cleaned up.
