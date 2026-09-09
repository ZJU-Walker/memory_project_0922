# v6 — token-level contextual keys for the sentence fast-weight bank

Started 2026-09-08 19:20 (user: "ok do it, and make sure we can recover to what we have now … call this v6").

## 0. Recovery

* v6 is branch `v6` of the same repository, worktree `/iris/u/kewalk/memory_project_v6`, created from v5 commit
  `9febb98` (= tag **`v5-task1-data-20260908`**, the v5 line with the task1 data prep and the token-key probe).
* Nothing in `/iris/u/kewalk/memory_project_v5` is touched by v6 work: its checkpoints, LeRobot datasets, configs and
  the B9 recipe remain the fallback. v6 reads the existing datasets through symlinks
  (`v6/data/lerobot/yam/{bean_scoop_0902_v5,bean_scoop_0905_v5,task1_find_0908_v5}` → v5) and writes everything new
  under `v6/` (checkpoints, diagnostics, assets, new datasets) and `openpi/cluster_v6/`.
* Every v6 model flag defaults OFF, so a v5 config loaded in v6 code behaves exactly as in v5 (tested by the unit
  tests listed in §4 before the first training run).
* Raw label files of the new convention get their own names (`subtask_labels_task1v6_<target>.json`), a new episode
  view (`data/0908_task1_episodes_v6`) and a new LeRobot repo (`yam/task1_find_0908_v6`); the v5 files stay.

## 1. Why (evidence, all bank-level with the B9-2000 encoder; `scripts/v5_token_key_probe.py`, report
`memory_project_v5/v5/diagnostics/token_key_probe_B9_2000/`)

The v5 bank writes ONE pooled key/value per sentence. Sentences of one shape collide (A8 measured cosine 0.996-0.999
for count-only differences), which A8 patched with template "slots" derived from the closed sentence list. On the
task1 vocabulary that rule gives **4 slots for 28 sentences** (all twelve `{obj} in bin {k}` in one slot): the four
placements would overwrite each other. Any per-task slot rule is exactly what the user does not want.

Token-level writes — every note token written under the CAUSAL contextual state that precedes it, value = the token
— are the addressing a fast-weight language model has for free, and they need no vocabulary or prompt knowledge:

| read at the decision point, same frame as the note | pooled sentence key (v5 pre-A8) | token-level causal keys |
|---|---|---|
| task1: bin of each object after all 4 placements (n=284) | 0.47 | **1.00** |
| beans: blink count before the go step (n=89) | 0.38 | **1.00** |
| beans: k and x at every tray arrival (n=152/151) | 1.00 | **1.00** |
| old bin: banana side after close-and-reset (n=70) | 0.51 | **1.00** |

Unchanged at the real bank size (512-d keys / 2048-d values, random projection) and under the real decay 0.01.
Conditions: keys must be causal (query consistency 1.000 vs 0.883 bidirectional), states standardized (raw = chance),
and standardizing the VALUES too lifts margins ~10x (task1 0.012 → 0.118, beans 0.019 → 0.232, old bin 0.066 → 0.71)
and removes late interference (task1 episode-end 0.81 → 1.00, beans 0.70 → 1.00).

**What is not free:** a question asked from a DIFFERENT frame ("all bins closed, banana is in bin ?") reads at chance
parameter-free (0.23-0.27; beans go-frame 0.70). The bank stores under the writing context; mapping a question to it
is the trained query projection every fast-weight LM has. Hence §3.

## 2. Label convention (general, not task-specific): decisions restate the remembered note

The closing sentence of task1 becomes the note itself, `banana in bin 2` (instead of `all bins closed, banana is in
bin 2`): the question is then posed in the note's own frame, the parameter-free read is exact (1.00) from the first
step of training, and the learned query has an easy target. Vocabulary: 16 sentences (`watching: no object placed
yet`, 12 × `{obj} in bin {k}`, 3 × `open bin {k}`). Placement notes stay prompt independent (the user wants the model
to remember everything, not only the target). Beans and the old bin task are unchanged (their decisions already read
in-frame or copy the newest note).

## 3. Model change (openpi/src/openpi/models/pi0.py, memory.py; all behind new flags, default off)

1. **Token-level writes** (`memory_v6_token_writes`): when the sentence changes, run the sentence through the
   memory-blind encoder CAUSALLY (mask_ar = 1; the v5 pass is bidirectional), take the layer-`memory_layer` states
   h_0..h_{n-1}; write n associations with `delta_write_kv_multi` (f = sentence length, mask = valid tokens):
   key_t = P_k · std(h_{t-1}) (t ≥ 1; key_0 = P_k · std(h_0)), value_t = P_v · std(E[x_t]).
   `std` = per-feature standardization against the reference token statistics (the A6 mechanism, recomputed under
   stop-gradient from the current blocks; task-agnostic, it only needs the sentence list the config already pins).
   P_k, P_v: the existing `memory_sem_key_proj` / `memory_sem_value_proj` shapes (2048 → 512 / 2048), initialised
   as in A6 (identity block for values, random-orthogonal for keys), trainable.
2. **Pointer read** (`memory_v6_pointer_read`): while the model decodes its sentence (causal region), at every
   position the decoder's layer-`memory_layer` state s is projected by a NEW trainable W_q (2048 → 512, init = P_k
   so the untrained read already matches same-frame questions), the bank returns r = read_key(state, q);
   the next-token logits get `+ beta · (E_std · r)` restricted to the reference token set (beta trainable scalar,
   init 1). Training signal: the ordinary sentence CE, which now flows into W_q, P_k, P_v and beta.
3. **Existing reads** (`v5_semantic_queries` → memory tokens in the prefix, used by the action expert) stay as they
   are; they now read token-valued content. The A8 slot/whitening flags stay available but are OFF in v6 configs.
4. **Write rule/decay** unchanged (delta rule, `alpha_step`, retry-until-committed, delay 0).

## 4. Tests before the first run (all must pass)

* `pi0_v6_test.py`: (a) with all v6 flags off, forward/loss bit-identical to v5 on a fixed batch; (b) causal token
  states: state t does not change when tokens > t change; (c) token writes + pointer read on a toy vocabulary
  reproduce the probe (four facts, newest-wins counting) inside the model; (d) shapes/masks for the multi-slot
  commit with variable sentence lengths.
* Bank-level probe re-run through the MODEL's write path (not the standalone script) on the three vocabularies.

## 5. Plan

1. Data: v6 labels (§2) → episode view → LeRobot `yam/task1_find_0908_v6` → `cluster_v6/task1/` manifest + sidecar.
2. Model: §3 + tests.
3. Stage A (label writes) on task1, battery: same-frame and cross-frame read accuracy on the development split
   (demo10, demo19, demo54) at every checkpoint; then stage B (own writes); then the robot.

## 6. Status log

* 2026-09-08 19:20 — worktree created; v5 tagged; data links in place; `uv sync --frozen` for the v6 venv running
  (`v6/uv_sync.log`).
* 2026-09-08 19:30-19:57 — **model + tests**: v6 methods in pi0.py/pi0_config.py (`memory_v6_token_writes`,
  `memory_v6_pointer_read`, `memory_v6_value_standardize`, `memory_v6_pointer_beta_init`); `v5_commit_sentence`
  dispatch used by the server and `v5_heldout_video.py`. `pi0_v6_test.py` 5/5 (tiny stand-in with a 32-d key /
  64-wide bank: four same-shaped facts side by side, newest wins, pointer bonus zero at init and only on reference
  tokens, sequence loss finite with non-zero gradients into beta and the token key projection).
  `pi0_v5_test.py` 26/26 — NOTE the v5 line itself fails `test_v5_oracle_sequence_writes_on_every_sentence_change`
  at tag `v5-task1-data-20260908` (`'_TinyV5Seq' has no attribute 'v5_sentence_kv'`: A8 added the method to the
  model and never to the tiny test class); v6 fixes the test class only. Two v6 tests were first written against
  the v5 tiny bank (8-d keys) and the 2-token sequence fixture and failed for those reasons, not for the model.
* 2026-09-08 19:46 — **data**: LeRobot `v6/data/lerobot/yam/task1_find_0908_v6` (71 episodes / 46,553 frames,
  16 sentences; 20 min on iris-hgx-1); `cluster_v6/task1/task1v6_episode_manifest_v1.json` sha256
  `5ade0b0d6e08692f760736ca50fc10f4bd7872e02f26efc7347105824eceeb46`, `task1v6_v5_subtask_labels_v1.json` sha256
  `4baf65762486e818c9110497797368ba171cb0e475978c42ace59b87480579a8`; development = demo10 (ep 12/13), demo19
  (26/27), demo54 (67/68). BUG found on the way: `task1_build_v5_manifest_sidecar.py` wrote the split RULE string
  as `split_seed`; the loader does `int(split_seed)` → the v5 file `cluster_v5/task1/task1_episode_manifest_v1.json`
  would crash a v5 task1 training (fix = rebuild it with `--split-seed 908`; v5 untouched).
* 2026-09-08 19:50 — **config**: `V6_TASK1_*` constants (reference rows verified against the tokenizer),
  `v6_task1_data` (beans-0905 loader settings, split seed 908), `pi05_yam_mem_v6_task1A` (oracle writes, token
  writes + pointer read, slot keys/whitening off, warm start beans B9 ckpt 2000, fresh `memory_v6_*`, 2000 updates,
  ckpt every 250) and `pi05_yam_mem_v6_task1B` (own writes, retry, half lr; stage-A params via
  `OPENPI_V6_TASK1_A_PARAMS`). `project_paths.SHARED_DATA_LINKS += "v5"` (top-level link `memory_project_v6/v5`
  → `memory_project_v5/v5`, read-only). `cluster_v6/env.sh` (worktree-local caches; `v35/cache/openpi/big_vision`
  copied, `openpi-assets` linked).
* 2026-09-08 19:48 — **GPUs** (user 19:47: "for test and training use 17315830 4h100, you can stop the memoryvla
  training there but keep the 1gb alive"): the MemoryVLA torchrun (4 ranks, `MemoryVLA/cluster/yam_beans0905/
  run_pipeline.sh memoryvla_beans0905_r1`, ~62 GB per GPU) was stopped with SIGTERM; the 1 GB `train_hs.py`
  keep-alive (pid 3743806) is untouched; the four H100 show 1013 MiB each.
* 2026-09-08 19:57 — norm stats running on iris-hgx-1 (`v6/logs/norm_stats_task1v6.log`); stage-A queue
  `cluster_v6/task1/queue_task1A_hgx1.sh` (norm stats → `run_train_hgx1.sh` 4 GPUs batch 8 → development battery
  `run_task1_evals_hgx1.sh` per kept checkpoint, GPU 0). Battery output `v6/diagnostics/videos_<exp>_<step>/`.
* 2026-09-08 20:05-20:50 — **first two launches**: (1) 20:04 died at start — `cluster_v6/env.sh` had moved
  `HF_LEROBOT_HOME` to `v6/data/lerobot`, train.py enforces the v3.5 contract value (fixed fe70c2d; the v6 dataset is
  reached through the config's explicit `lerobot_dataset_root`). (2) 20:06 loaded data (65 train episodes) and passed
  the weight audit (158 matched from beans B9-2000, 7 fresh = the four `memory_v6_*` leaves + 3 empty bias slots) but
  sat in XLA compilation for 44 min (v5 A9/B9 reached step 2 in ~10 min): `delta_write_kv_multi` unrolls its slot
  loop in Python and v6 passes f = 48 slots (padded sentence length) per write, inside the 40-step scan and 16x in
  the prefill. Fix: `slot_loop="scan"` (same per-slot math as one `lax.scan` body; every pre-v6 caller keeps the
  unrolled loop, bit-identical) — `test_v6_scan_slot_loop_matches_the_unrolled_loop`; memory_v4 + v6 suites 13/13.
  Relaunched 20:51 (queue re-armed; run dir → --overwrite).
* 2026-09-08 20:51-21:05 — launch 3 (20:51, scan fix) died 6 s after the data loader came up: NCCL `ncclGroupEnd`
  "Cuda failure 999 unknown error" on the first multi-GPU execution (GPUs healthy, a 4-GPU psum inside the job passed
  at 20:56 → transient after the SIGKILL of launch 2). Launch 4 at 20:56 (`cluster_v6/task1/restart_queue_hgx1.sh`):
  data loader 21:00:16, **Step 0 at ~21:05** (compile now ~5 min, as v5). Step 0: ce_loss 6.89 (new task, new
  sentences; B9 started at 2.4 on its own beans vocabulary), 19 token-level commits in the batch, v5 qk-cos 0.07
  (the pooled-key read queries have to adapt to token keys), grad_norm 33 (clip 1.0), memory_grad_norm 7.2 (clip 5),
  no NaN. Exp `v6_task1A_20260908_r1` = the fourth launch's run dir (--overwrite); status log has the 3 exit=1 lines.
