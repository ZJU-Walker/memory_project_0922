# v7 — phase context on top of the v6 bank (worktree created 2026-09-12 12:20)

## 0. Recovery

* v7 is branch `v7` of the same repository, worktree `/iris/u/kewalk/memory_project_v7`, created from v6 commit
  `be02522` (branch `v6` after the boba_0911 labels / base config commit). Tracked content only (76 MB): no
  checkpoints, datasets, caches, logs or diagnostics were copied (user 2026-09-12 12:16: "dont copy large files
  like ckpt and unnecessary stuffs").
* Read-only links (all untracked, recreate by hand if the worktree is re-made):
  `data -> /iris/u/kewalk/memory_project_v4/data` (raw demos, manifests), `v5 -> memory_project_v5/v5`,
  `v6 -> memory_project_v6/v6` (v6 datasets incl. `v6/data/lerobot/yam/boba_0911_v1`, the boba base checkpoint
  `v6/checkpoints/pi05_yam_boba0911_base/pi05_boba0911_base_rtc15_20260912_r1/{5000,9999}`, v6 assets/norm stats).
  `project_paths.SHARED_DATA_LINKS` now lists `v6`, so `project_path("v6/...")` resolves through the link.
* Caches: `v35/cache/{huggingface,uv,openpi}` are links to the v6 caches (HF arrow copies of the datasets, wheels,
  the pi05_base weights); only the JAX compilation cache (`v35/cache/jax`) is v7's own (never share it between
  concurrent JAX processes). `openpi/.venv` was created with `uv sync --frozen` (log `v7/uv_sync.log`).
* Every new v7 model flag must default OFF so a v6 config loaded in v7 code behaves exactly as in v6.
* Artefacts of v7 runs go under `v7/{assets,checkpoints,diagnostics,logs}` (`cluster_v7/env.sh`, `cluster_v7/train.sh`).

## 1. Why (user 2026-09-12 12:16)

The boba base policy (v6 `pi05_yam_boba0911_base`, 98% per-frame subtask accuracy on held-out episodes) is
"a bit hard to understand in each subtask that is the current phase": a single frame is ambiguous at phase
boundaries. Two candidate remedies to compare before the memory policy is trained: (1) bring the visual memory
back, (2) short-term memory = feed the last ~5 frames. See §2 for the analysis and the decision.

## 2. Analysis: visual memory vs. a short-term frame window (2026-09-12)

What the model sees today (per memory step, `pi0.py:5114-5121`): ONE frame per camera (3 x 256 SigLIP tokens),
the 14-D joint state (positions only, no velocities) and the sentence. The v6 bank is a language memory: keys and
values are text-derived, the visual bank of v3.x-v4 is still written but its read is zeroed
(`memory_v4_visual_injection=False`) and frozen at deployment (`write_mode="frozen"`). So NO past information of any
kind reaches the sentence head except through the sentence bank, and the only motion cue is what one frame shows.

Why one frame is ambiguous about the phase: the same arm pose occurs on the way to the bin and on the way back,
and joint positions carry no direction of motion. Boundary frames (lid just released, spoon just lifted) are
ambiguous by construction. On the boba base (`pi05_yam_boba0911_base`, stride 15, dev episodes) this costs
1-2 % of frames, all single-frame flips at boundaries and between `scoop 2 of 3` / `3 of 3`
(`v6/diagnostics/base_eval_pi05_boba0911_base_rtc15_20260912_r1_9999/`). At deployment the same flips become
spurious sentence commits (commit = sentence changed & confidence >= 0.9).

### Option 1 — bring the visual memory back: NOT recommended

* It is the wrong tool for "which phase am I in": a delta-rule fast-weight matrix (512 x 2048) with decay 0.01 per
  step stores a compressed associative summary of past top-camera features; it does not return "the frame 200 ms
  ago" and cannot express direction of motion.
* Five versions of evidence that its content was never used: injection RMS ~62,600x below the residual stream
  (v3.2/3.3), "zeroing retrieved memory usually did not change the selected side" (v3.3), v4 "the backbone reads
  'a memory is present' more than 'which fact' ... memory USE erodes with more training", plus the write
  saturation / backward-explosion incidents that made v3.4-v3.5 unstable (`memory.py:28-32, 106-123`).
* It re-opens the leak problem the sentence bank was built to close: a visual bank that stores the reveal frames
  is a second, uninterpretable route to the memory target, so a v7 result could no longer be attributed to the
  sentence bank. The visual subsystem is frozen in v5/v6 for exactly this reason.

### Option 2 — short-term window: right idea, wrong size

* A literal 5 frames x 3 cameras = 3840 image tokens, 3.85x the current 1080-position step; inside the 40-step
  memory scan this is out of reach (v6 already OOM'd at global batch 8 on 2 x H100 at 1080). Deployment latency
  scales the same way (currently ~305 ms per note request).
* The cheap variants that give the same information:
  (a) **state history**: append the joint state of the previous 1-2 memory steps (or the velocity) to the
      "State:" prompt, +14-28 tokens. Direction of motion for every arm phase, zero image cost; the loader's
      `delta_timestamps` machinery (`data_loader.py:208-221`) already fetches multi-offset state. Cannot leak the
      count or the reveal (a 0.3 s window).
  (b) **previous sentence in the prompt** (BEANS_LABELS baseline (b), never implemented): the phase becomes an
      input and the model only detects transitions; +~12 tokens. Caveat: with "k of x" labels the previous
      sentence carries the count, so the memory experiment must define what the bank alone has to supply
      (for boba: the count is visible anyway; for a shuffled-layout test the bank supplies the bin).
      `memory_v5_query_prev_sentence` already feeds it into the read query, not the prompt.
  (c) **compressed visual history in the 16 existing slots**: fill the zeroed visual-memory positions with the
      `write_query_compressor` output (16 tokens) of the previous frame(s) instead of a fast-weight read — a
      recency window, no delta rule, no decay, no extra sequence length; needs a 16 x 2048 ring buffer on the
      server. Only worth it if (a)+(b) leave visual-only ambiguities (e.g. spoon empty vs loaded).
* Zero-cost first step: temporal smoothing of the sentence at the server (commit only after k consecutive
  identical decodes) removes single-frame flips without any training.

### Decision (proposed)

Implement (a) and (b) in v7 as flags defaulting OFF, run the boba base with them at stride 5, compare per-frame
subtask accuracy and flip counts on the dev episodes against the plain base; add (c) only if visual ambiguities
remain. Keep the visual fast-weight bank off. Then train the memory policy on top of the better variant.
