# Template-slot SNAP ablations

The [main README](../../README.md) is the current runbook: stop/update/archive commands, the four main rows, and A250→B3000.
Historical token-bank recipes are not interchangeable with this recipe (`template_slot_ab_v1`).

## One tick

1. Read every automatically induced sentence-template address from the sentence fast-weight bank. Current vocabulary: 5.
   Occupancy masks hide unwritten addresses. Retrieval is parallel, not sequential token decoding.
2. Inject those vectors at the existing input-layer interface (after the prompt, before transformer blocks).
   Memory tokens keep their existing blind-attention contract; the language/action streams may use them.
3. Decode the sentence/actions. A writes the label; B trains its writer with teacher-forced argmax; deployment decodes freely.
4. Encode the whole committed sentence as the value. Its template encoding is the key. Update using the delta rule and
   one decay step. Writes require a known reference sentence, a changed note, and minimum token probability 0.3.
   One-tick debounce preserves short light events. No token pointer or 48-token read-back exists in these configs.

Keys and values keep the B9 text-only encoder/pooling/whitening design. Read keys use the same reference computation as
write keys. Direct read removes the need to learn which template address to ask for; it does not remove cross-talk,
sentence-generation errors, or the transformer's need to learn how to use retrieved values.

Template induction groups equal-length training token sequences connected by at most two differing positions, then masks
all variable positions per connected component. Distinct retained token sequences become addresses. A new task supplies a
new sentence vocabulary and re-runs this procedure; groups must be inspected because this heuristic is not universal
semantic schema induction. There is no hand-coded wait/light/go/scoop/done router.

## Auxiliary bank

- `vis8`: eight learned pooling queries over the front camera's memory-blind image tokens produce eight write vectors.
- `state8`: normalized proprioceptive state passes through a learned projection, producing one write vector.
- `vis8s`: both sources write into one shared bank (eight visual + one state association per tick).
- All three read eight learned-query vectors, appended after the five sentence vectors. Gates use the existing fixed 0.5
  initialization; auxiliary injection is scaled to image-token RMS.
- Main rows use delta updates; optional `*_add` rows use additive updates in this bank only.
- Inputs to the auxiliary writer are stop-gradient; its pooler/projections still learn through the in-window sequence.
- Training first replays past observations chronologically with no gradient/loss. Negative offsets exclude the sampled
  window; clamped pre-episode padding neither writes nor decays. Capacity 320 ticks covers the current data and fails
  explicitly if a start frame exceeds it. Increase it for longer tasks.
- Offline full-episode evaluation now carries this bank across ticks; serving already did so. `--vis-zero-read` is the
  matched reliance intervention. The sentence-bank reliance switch remains available separately.

## A/B and comparability

Each row: KI base/10000 → own A250 (oracle writes) → own B3000 (predicted-content writes). B loads **all** A parameters,
with fresh optimizer, rather than accidentally fresh-initializing memory. Both stages retain label sentence-history
prefill; B is not purely free-running autoregressive training. No 500-step write-label ramp.

Common: 4 GPUs, seed 42, v4e onset sampling/losses, lr 2.5e-5, 40 ticks, 5-frame stride, TBPTT25, decay 0.999/tick.
User-selected batches: H200 `BATCH=16 ACCUM=1`; H100 `BATCH=12 ACCUM=1`. Both run without accumulation.
Equal steps therefore expose H200 to 4/3 as many training windows; report this confound in cross-cluster comparisons.
No automatic OOM batch reduction. A/B have separate config/experiment names, with per-run recipe guards.
Auxiliary history is expensive (video decoding and frozen image encoding); measure throughput before large sweeps.

## Files and checks

| file | purpose |
| --- | --- |
| `run_<row>.sh` | unchanged user-facing row entrypoints |
| `run_stages.sh` | recipe guard, per-row A→B and resume |
| `train_slot_stage.sh` | one stage, direct or Slurm; allows 1 GB keep-alives |
| `train_ablation.sh` | legacy launcher retained for historical reproducibility; not used by new wrappers |
| `ablation_ctl.sh`, `manage_runs.py` | checkout-scoped status/stop/recoverable archive |
| `run_tests.sh` | CPU regressions; A2→B2 real-data smokes; optional short probes |
| `00_download.sh`, `setup_other_cluster.sh` | unchanged dataset/base/cache setup |

Tests cover template discovery, read/write address equality, unwritten masking, unknown rejection, finite sequence gradients,
stage parity, auxiliary row differences, past-only replay and padding, and the existing bank/serving contracts.
Logs retain normal telemetry keys: `vis_bank_norm`, `vis_read_rms`, `vis_read_injected_rms`, `vis_commit_rate`,
`memory_grad_norm`, sentence/flow losses.
