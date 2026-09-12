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

## 3. Boba memory line on the 2xH200 (2026-09-12 13:16, user: "launch training in my 2h200 ... start from our already trained base pi05 boba ckpt")

Configs `pi05_yam_mem_v7_bobaA` / `pi05_yam_mem_v7_bobaB` (config.py, block "v7 boba memory line"), built by
`_v7_boba_mem_variant` on top of the v6.1 A2 recipe (linear delta-rule sentence bank, token-level causal keys whitened
over the reference vocabulary, context-query pointer read with beta init 10, analytic history prefill, semantic-only
freeze). Differences to v6 task1, all deliberate:

| knob | v6 task1 | v7 boba | why |
|---|---|---|---|
| warm start | beans B9 / task1 A ckpt | boba base 9999 (`v6/checkpoints/pi05_yam_boba0911_base/.../9999/params`) | every `memory* / fact_* / query_* / state_null_embedding / probe_head / ladder_*` leaf fresh, the rest matched (audited loader) |
| memory stride | 5 frames | 15 frames (2 Hz) | user 12:5x "lets do 2hz" (stride viewer) |
| window / buckets / fence | 40 / 14,27,40 / 25 | 60 / 20,40,60 / 30 | episodes are 300-415 steps at stride 15; 60 steps = 30 s covers a whole scoop cycle x3; the fence only cuts the gradient (VRAM unchanged) |
| prefill buffer | 16 | 22 | an episode has at most 21 distinct sentences before a window (kept: the most recent) |
| full-trajectory mass | 0.25 | 0.05 (`memory_slice_prob` 0.9) | a "full" window is only the first 60 of ~300 steps |
| simulated RTC delay | 6 | 15 | same budget as the base |
| state masking | 0.5 | 0 | the v3.4 anti-leak measure removes proprioception on half the decision windows; nothing leaks here and the base never saw it |
| decision / evidence sets | task1 bins | scoop 2/3 of 3, put the scoop back (x2), second get cup / open bean bin; watch + scoop sentences | telemetry only in generic mode (`v4_decision_ce`); the sentence CE grades every step |

Data: `cluster_v7/boba/boba_episode_manifest_v1.json` + `boba_v5_subtask_labels_v1.json` (schema v1, SHAs pinned in
config.py) from `scripts/boba_build_v5_manifest_sidecar.py` over the converted `yam/boba_0911_v1` and the converter
manifest `data/0911_boba_episode_manifest_v1.json` (split copied: 50 train / dev demo11,19,37 / final_test demo13,41,59;
seed 911). Reference tokens = PaliGemma ids of `sentence.lower().strip() + "\n"` (21 rows).

Stages (chain `cluster_v7/boba/chain_mem_hgx2.sh`, job 17403858 = 2xH200 on iris-hgx-2, FSDP 2, global batch 4 =
2 windows of 60 steps per GPU, 16 CPUs / 12 workers; the job's 1 GB `train_hs.py` keep-alives stay):
* **A** `v7_bobaA_20260912_r1`: oracle (label) writes, 500 updates, lr 5e-5 flat after 100 warmup, checkpoint 500 kept
  (250 transient).
* **B** `v7_bobaB_20260912_r1`: own writes (retry-until-committed, label content), from A/500, lr 2.5e-5, 3000 updates,
  saves every 500, keeps 1500 and 3000 (disk: 617 GB free at launch, ~40 GB per checkpoint; the three pending ctx runs
  were switched to keep only their final checkpoint for the same reason).
Logs `v7/logs/train_v7_boba{A,B}_20260912_r1{,_status}.log`, `v7/logs/chain_mem_hgx2.{out,log}`.

**r1 -> r2 (15:56 / 15:58).** The user cancelled the 2xH200 job 17403858 at 15:56 to get the 4xH200 job 17403682 ("lets switch
to 4h200 version to make training faster"); stage A r1 died at update 235 (28 s/update on 2 GPUs; CE 2.67 -> 1.07 and
decision exact-match 96 -> 98 % by update 200; no checkpoint yet, the first save was at 250). r2 = the same recipe and
global batch 4 on FSDP 4 (`cluster_v7/boba/chain_mem_hgx2_4gpu.sh`, one 60-step window per GPU, CPUS 24), exps
`v7_bobaA_20260912_r2` / `v7_bobaB_20260912_r2`, logs `v7/logs/train_v7_boba{A,B}_20260912_r2*.log`,
`v7/logs/chain_mem_hgx2_4gpu.{out,log}`. Batch 8 (two windows per GPU, the r1 VRAM footprint) would fit too but
would not shorten an update; the point of the switch was wall-clock.

Runtime caches: the memory path runs `configure_v35_runtime_environment`, which rejects any `v35/cache/*` entry whose
resolved path leaves the v7 tree (first launch 13:33 died on the symlinked `v35/cache/uv`). `v35/cache/{uv,openpi,
huggingface}` are therefore REAL directories since 13:36: `uv` empty, `openpi/big_vision` a copy of the tokenizer,
`openpi/openpi-assets` a link to the v6 copy of pi05_base (not needed by any v7 config; a gs download through it would
still fail closed), `huggingface/{hub,modules}` copies and `huggingface/datasets/parquet/default-e295320b2b6e3ab9` the
same link to the `~/.cache` arrow copy v6 uses (217 GB; the hash is the dataset root's, identical for v6 and v7).

Evaluation (next): `scripts/v5_heldout_video.py --config-name pi05_yam_mem_v7_boba{A,B} --params <ckpt>/params
--episode-index <dev idx> --write-mode {self,oracle} --manifest cluster_v7/boba/boba_episode_manifest_v1.json --sidecar
cluster_v7/boba/boba_v5_subtask_labels_v1.json` on the dev episodes (LeRobot indices of demo11/19/37 from
`meta/episode_sources.json`), as in `cluster_v6/task1/run_task1_evals_v2_hgx1.sh`. The phase-context flags of §2 are
NOT yet ported into the memory path (config.create raises NotImplementedError for `use_memory` + those flags); port the
winning ctx variant once the ablation on the 2xH100 finishes.
