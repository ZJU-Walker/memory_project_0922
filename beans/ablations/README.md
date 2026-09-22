# beans0922 ablations — how the rows work

Everything here is one gated model flag, `Pi0Config.memory_vis_bank`, plus two switches for what feeds it and one for the
update rule. With the flag off the model is snap, bit for bit (tested). The top-level README has the row table and the
commands; this page explains the mechanism so the rows can be read.

## Snap in one tick

1. Prefix: 3 cameras × 256 image tokens + 80 prompt tokens + **8 sentence-memory tokens**, all at the input of the 18-block
   language model. The memory tokens are appended after the prompt; they attend only to each other (blind), everything else
   may attend to them.
2. Read: 8 fixed learned queries → the sentence bank W (a 512 × 2048 matrix per episode) → 8 vectors → tanh gate + RMS
   matched to the word embeddings + slot embedding = the 8 tokens. A fresh bank reads exactly zero and the tokens are masked.
3. The model decodes its sub-task sentence (up to 48 tokens) and the action chunk.
4. Write: each word of the sentence becomes an association (key = the memory-blind context of the words before it,
   value = the word's embedding); committed with the **delta rule**, then the whole bank decays by 0.99. In training the
   label sentence is written instead of the model's own with probability 1 → 0 over the first 500 updates.

## The sensory bank (rows vis8 / vis8s / state8 and their `_add` twins)

A second bank, same size and same read mechanism, next to the sentence bank:

- **Read** — 8 more fixed learned queries → the sensory bank → tanh gate (same 0.5 init) + RMS matched to the sample's own
  image tokens + slot embedding → **8 more input tokens** after the sentence tokens. Empty bank ⇒ exactly zero, masked,
  so tick 0 equals snap. Nothing else about the model changes (the sentence/action tokens simply sit 8 positions later).
- **Write, every valid tick**, from memory-blind, stop-gradient inputs so the bank can never store what it read:
  - *image slots* (`memory_vis_image_write`): the front camera's 256 input image tokens (SigLIP + projector) are pooled by 8
    learned queries into 8 vectors v₁..v₈; slot i is stored as key = unit(P_k v_i + e_i), value = unit(P_v v_i);
  - *state slot* (`memory_vis_state_slot`): the normalised 14-D joint state through a fresh linear map φ; key =
    unit(P_k φ + e_state), value = unit(P_v φ).
  P_k, P_v, the pooling queries and the slot offsets e are trained; keys of one tick land in distinct directions because
  of the offsets. In training the bank is only as old as the window (40 ticks = 6.7 s); at serving it runs from the first
  tick of the episode. Unlike the sentence bank it cannot be pre-filled from labels for windows that start mid-episode.
- **Update rule** (`memory.commit_rule`), applied slot by slot within the tick, then one decay step:
  - `delta` — W ← ρW + η (v − W k) kᵀ: only the error is written. A repeated association adds ≈ nothing: a *presence*
    memory ("this was seen"), with error correction between different items. This is also the rule of the sentence bank
    and of gradient-based test-time-training layers (RoboTTT-style).
  - `additive` — W ← ρW + η v kᵀ: the value itself is written every time. Repeats accumulate (three identical writes read
    back as (1 + 0.99 + 0.99²) v): a *tally* memory ("how often"), at the price of small cross-talk between items. The
    read token is RMS-normalised, so the network sees relative strengths, not absolute counts.
- The old visual bank of the v3 line is the same parameter slot (`model.memory`), reconfigured as the linear delta bank the
  sentence bank uses; its old layer-8 compressors stay inert. The v3.5 telemetry still sees snap's zero write, so the losses
  and side probes are unchanged.

## Recipe (identical for every row)

Warm start from `beans0922_base/10000` (pi0.5 + knowledge insulation, trained on the same data) with fresh memory
parameters; 3000 updates; label-write probability 1 → 0 over the first 500; lr 2.5e-5 constant after a 100-step warm-up;
FSDP over 4 GPUs; batch = the largest that fits (launcher default 16, fallback 12 / 8 / 4); checkpoints every 250 (two newest
kept for resume), 1000 / 2000 / 3000 permanent (~27 GB each; `OPENPI_BEANS_AB_KEEP=500` for a finer grid); W&B project
`beans0922_ablation`. Tick 5 frames, 40-tick windows, TBPTT 25, the 20 target-carry sentences, no
state masking — all snap's (`openpi/src/openpi/training/beans0922_config.py`).

## Telemetry to watch (W&B `diagnostic/`)

`vis_commit_count` (should equal the number of valid ticks), `vis_bank_norm_sum` (bounded; grows towards a plateau under
the additive rule), `vis_raw_read_rms_sum` (the raw retrieval before the gate), `vis_injected_pre_cast_rms_sum` (after the
gate; ≈ gate × image-token RMS × ticks), next to snap's usual sentence/flow losses and memory-group gradient norm.

## Tests

`openpi/src/openpi/models/pi0_v0922ab_test.py` (tiny model): flag off == snap bit-for-bit; empty bank reads zero; the write is
stop-gradient and trains only `memory_vis_*`; commit / exact-decay / invalid-tick contract; a second association does not
destroy the first; sequence loss finite with gradients to every sensory leaf; zeroed read cannot leak the write; the sampler
advances the bank in "normal" mode only; additive accumulates and delta does not; the state slot depends on the state only;
state-only rows ignore the images. `openpi/src/openpi/training/beans0922_ablation_test.py`: every row differs from the
control in exactly the intended fields. Serving: `openpi/scripts/serve_yam_memory.py` advances the sensory bank once per
served tick; `--vis-zero-read` silences its read for a reliance test.

## Node-local data (speed) and the two path guards

The loader memory-maps its arrow cache; over NFS that stalls training (measured: 2 s/update vs 1.8 updates/s local). The
tree's path guards only accept in-project paths, with two sanctioned exceptions for node-local disks: symlinks at or below
`v35/cache/` (caches) and entries of `local/` (mirrors of project data). `00_download.sh LOCAL_DISK=<dir>` creates both
links (`local/bean_scoop_0905_v5` and `v35/cache/huggingface/datasets`); `train_ablation.sh` uses `local/bean_scoop_0905_v5`
automatically when it exists. Do not point `OPENPI_BEANS_DATASET_ROOT` or `HF_DATASETS_CACHE` at a raw outside path for a
memory run: the v3.5 authorization refuses it.

## Files

| file | purpose |
| --- | --- |
| `setup_other_cluster.sh` | one-shot setup on another machine: clone + `uv sync` + `00_download.sh` |
| `00_download.sh` | once per machine: dataset + norm stats + the published base checkpoint (step 5000 until 10000 lands; re-run to upgrade) from the Hub, tokenizer caches; `LOCAL_DISK=<dir>` keeps dataset + cache on the node's disk |
| `train_ablation.sh` | generic 4-GPU launcher (waits for the base checkpoint and free cards, OOM fallback ladder, resume) |
| `run_snap.sh`, `run_vis8.sh`, `run_vis8s.sh`, `run_vis8s_add.sh`, `run_state8.sh`, `run_state8_add.sh` | one row each: `[GPUS=0,1,2,3] bash beans/ablations/run_<row>.sh [smoke]` |
| `ablation_ctl.sh` | `status` / `stop [<row>]` |
| `hf_upload.py` (+ `_node.sh`) | the one-off pushes to the Hub (dataset done 2026-09-22; base pushed when 10k lands) |
