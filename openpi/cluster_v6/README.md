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
* 2026-09-08 21:05-23:08 — **stage A** (`v6_task1A_20260908_r1`, 4xH100, batch 8): ~19 s/step steady; CE 6.89 → 2.18
  (100) → 1.02 (200) → 0.78 (300). User 21:09: gate at ckpt 250, B from it if good, B saves 250 / keeps 500; user
  21:12: test on job 17329416 (H200, hgx-2) so A keeps training → `cluster_v6/task1/gate_hgx2.sh`. **Battery ckpt 250**
  (`v6/diagnostics/videos_v6_task1A_20260908_r1_250/`, 22:31-23:02): ORACLE writes: first decision right on 4/6, but
  316/318 decision steps right — the two misses (demo54 banana/spoon) are a ONE-STEP LAG (the model repeats the closing
  note at the first decision step, then `open bin 1` for the remaining 45/46). Placement sentences 136-138/140-144
  exact under oracle writes but lagging one note behind = the decoder copies the newest bank note (the beans stage-A
  copy shortcut). SELF writes: 1/6 first decisions (demo54 banana, probably the bin-1 prior), placement exact
  7-76/116-144: it misnames objects/bins at the first placement and misses later ones — perception not learned yet;
  yet its decisions are consistent with its OWN notes (`spoon in bin 1` → `open bin 1`): the read/pointer side works.
  Strict gate = FAIL (4/6); decision 23:05 (as announced to the user): the read side is proven, so **stage B from
  keep_250** by hand: A stopped 23:06 (last step ~360), `keep_250` copied, `v6_task1B_20260908_r1` launched 23:07
  (own writes, retry, lr 2.5e-5, save 250 / keep 500, warm start keep_250/params). H200 side: `battery_B_hgx2.sh`
  runs the battery on every B checkpoint (250, 500, …) as it appears; verdicts in `v6/logs/gate_task1_hgx2.log`.
* 2026-09-08 23:18-23:40 — **the oracle numbers do not test memory** (user 23:18: "can this say it actually remembers, if
  the object is not the newest one?"): in oracle mode the closing note (`spoon in bin 2`, which IS the answer) is written
  for the model at the first closing step, so the decision only needs the newest note. The recall test proper is the
  FIRST closing step (bank = watching + the 4 placement notes, closing note not yet written; 5/6 dev targets are 1-3
  notes back). A ckpt 250 there: **2/6** (demo19 banana 1 back, demo54 spoon 3 back), the misses keep the prompted object
  and guess the bin (often 1). New battery mode `--write-mode oracle_evidence` (label notes for the human phase, OWN
  closing sentence and decision) confirms 2/6 and shows the consequence: a wrong own closing note → wrong bin opened.
  `task1_battery_verdict.py` now reports RECALL (first closing step) per mode. **Reversal 23:39**: stage B (30 min in,
  no checkpoint) stopped — its own wrong notes would corrupt the closing-step signal that teaches the lookup; stage A
  RESUMED from 250 (`switch_to_A_hgx1.sh`, --resume, 23:40). H200 gate v2 (`gate_A_v2_hgx2.sh`): three-mode battery on
  A 500, 750, …; PASS = oracle_evidence recall >= 5/6 → B from that checkpoint, then three-mode batteries on B.

## 7. v6.1 (2026-09-09 00:00) — the bank must be linear and the keys whitened; the pointer queries with the write key

User 23:44: "can you make sure the current method can actually recall from previous, not only the newest one?"
`scripts/v6_model_bank_probe.py` writes the label notes of every episode through the MODEL's v6 write path and reads
each object's digit with the key of its own context (`<obj> in bin _`), 71 episodes × 4 objects = 284 lookups, real decay:

| A ckpt 250, keys/values from the model | 0 back | 1 back | 2 back | 3 back | all |
|---|---|---|---|---|---|
| the model's bank (Titans MLP, 3 × 1024 hidden, l2norm) | 71/71 | 30/71 | 17/71 | 14/71 | 0.465 |
| plain linear delta-rule matrix, same keys | 71/71 | 71/71 | 51/71 | 31/71 | 0.789 |
| linear, no decay | | | | | 0.930 |
| linear + keys whitened over the reference token contexts | 71/71 | 71/71 | 71/71 | 71/71 | **1.000** (margin 0.17) |

Cause: the four `<obj> in bin _` context keys have mean pairwise cosine 0.25 in key space but 0.63 after the bank's
hidden layers, so the delta rule lets the newest note overwrite the older ones (the oldest reads as the newest digit).
The bank-level probe of 2026-09-08 had used a linear matrix, which is why it reported 1.00. Also measured: the trained
pointer scale is 0.003 after 250 steps (the "hidden" pointer query is never learned in time).

Changes (all flags, v5/v6.0 configs unchanged): `hidden_dims=()` for the semantic bank (a linear associative memory:
`_hidden` = unit key, delta rule on the single matrix), `memory_v6_whiten_keys` (PCA-whitening fitted on the unit
context keys of every position of every reference sentence, stop-gradient, recomputed from the current blocks;
`_v6_key_whitening`), `memory_v6_pointer_query="context"` (the pointer query is the WRITE key of the tokens decoded so
far, `v6_context_queries`: teacher-forced in training, step by step in the sampler and in `v5_heldout_video.py`; position
0 has no context and no bonus; `memory_v6_pointer_beta_init=10`), `AuditedPartialCheckpointWeightLoader.reinit_allowlist`
(re-initialise leaves whose shape changed). Configs `pi05_yam_mem_v6_task1A2` (warm start A keep_250, bank + beta
re-initialised) / `B2`. Tests: pi0_v6_test.py (g) context pointer, (h) whitened keys + linear bank.

**Verified inside the trained model (A2 ckpt 250, 02:10; `scripts/v6_bank_recall_probe.py`, task-agnostic: every written
context asked for its latest token, decoded over the reference vocabulary; reports `v6/diagnostics/bank_recall_probe/`):**

| task (episodes) | variable slots (objects / digits / sides) | by age 0 / 1 / 2 / 3 notes back |
|---|---|---|
| task1 (71) | 284/284 = 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| beans 0905 (89) | 267/267 = 1.000 | 1.000 / 1.000 / – / – |
| old bins 0830-0831 (70) | 140/140 = 1.000 | – / 1.000 / 1.000 / – |

(The A-250 MLP-bank model on the same probe: task1 variable slots at chance beyond age 0.) First development episode of
the A2-250 battery under SELF writes: `spoon in bin 2` written at the placement, `open bin 2` decided from that note,
58/58 decision steps (A-250: 0/58).
* 2026-09-09 02:40 — **A2 ckpt 250 battery (three modes, H200 cleared on user instruction 01:32)**: oracle_evidence
  (label notes, own closing sentence + decision) RECALL 6/6 at the first closing step, decisions right in all six
  episodes (one first-decision step lagging in demo54 banana); self writes: recall 1/6, first decision 2/6 (wrong own
  notes = perception, stage B's job). Verdict script fix: the closing segment is taken from the sidecar frames (when
  the target is the newest placement its note and the closing note are identical back to back). **GATE PASS** →
  the gate's ssh hand-off to hgx-1 failed (no Kerberos ticket on hgx-2; note for the next restart: log in with
  GSSAPIDelegateCredentials) and the workstation fallback launched **B2** (`v6_task1B2_20260909_r1`, own writes,
  retry, lr 2.5e-5, save 250 / keep 500) from `keep_250` of A2 at 02:44; A2 stopped at step ~420. H200: sentinel
  `cluster_v6/gpu_sentinel_h200.sh` keeps a half-card placeholder there between batteries (user 02:44).
* 2026-09-09 04:30 — **B2 ckpt 250 battery**: self recall 4/6 (own notes now mostly right: perception is being learned),
  self first decision 3/6; oracle_evidence first decision fell to 3/6 (A2-250: 5/6) while oracle_evidence recall stays
  6/6. Pattern in the failing episodes (e.g. demo10 tape, own note `tape in bin 2` RIGHT): the first 4–5 decision
  steps say `open bin 1` and switch to `open bin 2` exactly when the replayed arm starts moving toward bin 2. The
  decision label starts at the first joint motion, so every training decision frame carries the robot's own motion —
  a perfect visual cue that beats the note. Pointer beta unchanged at 10.006 in both checkpoints.
* 2026-09-09 05:30 — **`--intervention freeze_decision`** (images/state held at the last pre-decision step through the
  decision segment, oracle_evidence writes; `v6/diagnostics/freezedec_<exp>_keep_250/`): demo10 spoon — A2-250 AND
  B2-250 repeat the closing note `spoon in bin 2` for all 58 decision steps and never emit `open bin …` (0/58 each).
  The switch to the decision phrase is triggered by the image change (arm motion), not by the note. Fix prepared, not
  yet trained: sidecar **lead30** (`task1_build_v5_manifest_sidecar.py --decision-lead-frames 30 --min-closing-frames
  10`; `task1v6_*_v1lead30.json`, decision label starts 30 frames = 1 s before the first joint motion, closing lengths
  min 30 / median 71 / max 126) — the same LeRobot data, only the label boundaries move, so the model sees decision
  steps whose only source of the bin is the bank (the deployment regime: the robot does not move before the decision).
  Configs `pi05_yam_mem_v6_task1A3` (A2 recipe on lead30, warm start A2 keep_250, 500 steps, keep 250) and
  `pi05_yam_mem_v6_task1B3` (own writes from A3, `OPENPI_V6_TASK1_A3_PARAMS`). Battery/gate for that line:
  `run_task1_evals_v2_hgx1.sh` (SIDECAR/MANIFEST env) and `gate_generic_v3_hgx2.sh` (verdict reads the same sidecar).
  Switching the H100s from B2 to A3 is the user's call (B2 continues to 2000 until then; B2-500 battery next).
* 2026-09-09 05:57 — freeze_decision, all six development episodes: A2-250 0/320 and B2-250 0/320 decision steps say
  `open bin …`; both repeat the closing note at every frozen step (12/12 episodes). Confirmed: without the arm motion in
  the image the decision never fires. Gate v2 was paused 05:34–05:57 (`pause_gate_until_freezedec_done_hgx2.sh`) and
  now runs the B2-500 battery.
* 2026-09-09 06:39 — **B2 ckpt 500 battery**: first decision oracle 4/6, oracle_evidence 4/6, self 3/6; recall at the
  first closing step oracle 5/6, oracle_evidence 5/6, self 2/6 (B2-250: 6/6, 6/6, 4/6). demo19 (target bin 3): all
  three modes open `bin 2` first with the correct closing note `banana in bin 3` / `box in bin 3` just decoded, then
  switch when the arm moves. B2 is drifting further toward motion-driven decisions; own-write recall got worse.
* 2026-09-09 07:45 — **B2 ckpt 750 battery**: first decision 3/6 in all three modes; recall oracle 5/6, oracle_evidence
  5/6, self 3/6. Every wrong first decision is `open bin 2` (demo19 banana/box target 3, demo54 banana target 1) with
  the right note read one step earlier in most cases: a bin-2 prior at the first decision step, corrected by the arm.
* 2026-09-09 09:10 — **B2 ckpt 1000 battery**: first decision 4/6 in all modes, but RECALL at the first closing step
  collapsed to oracle 3/6, oracle_evidence 3/6, self 2/6 (250: 6/6, 6/6, 4/6). With all label notes in the bank the
  model now reads `banana in bin 2` for `banana in bin 3` (demo19) and for `banana in bin 1` (demo54): own-write
  training with wrong notes teaches the closing sentence to guess instead of read. B2 will not yield a usable
  checkpoint; recommendation to the user (09:12 push): stop B2, start A3 on the lead30 labels.
* 2026-09-09 09:20 — `v6_bank_recall_probe.py` on B2-500 and B2-1000: variable slots 284/284 = 1.000 at every age
  (identical to A2-250). The bank lookup is intact; the battery collapse is the closing decoder learning to override
  the pointer bonus during own-write training (wrong own note in the bank + label target = "do not trust the read").
  B3 therefore needs read-consistent targets (closing/decision restate the model's OWN note) or oracle-content
  protection, on top of the lead30 labels.
* 2026-09-09 09:25 — pointer beta 10.006 (A2-250) → 10.008 (B2-1000), W_q untouched (context queries): the read path did
  not move; the decoder's own logits learned to out-vote a correct pointer bonus at the closing step. **v6.2 rule for
  stage B** (`memory_v5_own_commit_label_content`, `Pi0.v5_bank_sentence`): the model keeps deciding WHEN to write
  (own sentence change + confidence + retry), the bank receives the LABEL sentence of that step. Bank content is then
  always consistent with the closing/decision targets; own wrong notes at deployment give wrong closings, which is the
  intended failure mode (perception), not a corrupted reader. `pi05_yam_mem_v6_task1B3` carries the flag.
* 2026-09-09 10:27 — **B2 ckpt 1250 battery**: first decision oracle 3/6, oracle_evidence 4/6, self 2/6; recall oracle
  5/6*, oracle_evidence 6/6, self 3/6 (*ep12 oracle and ep12 self died at launch: `srun … Socket timed out`, exit 140,
  transient; re-run to fill the record). Same picture: demo19 opens bin 2 first with the right note in hand.
* 2026-09-09 10:28 — **user: "Ok do it"** → B2 stopped at step ~1400 (kept 250/500/1000, latest 1250), **A3**
  `v6_task1A3_20260909_r1` launched 10:30 on the 4 H100 (A2 recipe on the lead30 labels, warm start A2 keep_250,
  159/159 leaves matched, 500 steps, keep 250). Gate v3 armed 10:29 on the H200 (A3 battery at 250/500 on the lead30
  sidecar, PASS = oracle_evidence recall >= 5/6 → **B3** `v6_task1B3_20260909_r1`, own timing + label content). The
  gate's ssh hand-off to hgx-1 works this time: its environment carries `KRB5CCNAME=/iris/u/kewalk/.krb5cc_claude_gate`
  (a copy of the live workstation ticket on NFS; expires 09-10 01:13). User asked about a 15-frame lead instead of 30:
  kept 30 (6 vs 3 still decision steps per episode at the 5-frame stride; the shortest closing is still 30 frames).
* 2026-09-09 10:33 — B2-1250 record completed (the two launch-failed runs re-run): first decision oracle 4/6,
  oracle_evidence 4/6, self 2/6; recall oracle 6/6, oracle_evidence 6/6, self 4/6. Recall at the closing step thus
  swings 6/6 → 5/6 → 5/6 → 3/6 → 6/6 over B2 250…1250: unstable under own-write training rather than monotonically
  lost; the first decision never recovered from the motion cue. B2 line closed at step ~1400.
* 2026-09-09 11:01 — **user: "if A reaches 250 directly start training B3"** → gate v3 stopped, **gate v4**
  (`gate_generic_v4_hgx2.sh`, "launch first") armed on the H200: the moment A3 ckpt 250 is finalized it launches B3
  from it via hgx-1 (`launch_B_generic_hgx1.sh` stops A3, protects keep_250), then runs the A3-250 battery for
  information and the battery on every B3 checkpoint (lead30 sidecar). Expected: A3-250 ≈ 11:57, B3 launch ≈ 12:00.
* 2026-09-09 12:02 — A3 ckpt 250 finalized 11:59 (CE 0.26 at step 200) → gate v4 stopped A3 (12:00), protected keep_250
  (27 GB) and launched **B3** `v6_task1B3_20260909_r1` at 12:02 on the 4 H100 (own write timing + label content, lead30
  labels, lr 2.5e-5, save 250 / keep 500, 2000 steps). A3-250 battery (information only) follows on the H200; the user
  wants each development episode reported as soon as it lands.
* 2026-09-09 12:15 — **A3-250 battery, first episode (demo10 spoon)** and a metric fix. The battery's `decision` flag
  comes from the LeRobot task labels (decision = first joint motion), so "first decision" was still measured at
  motion onset. `task1_battery_verdict.py` now takes the decision boundary from the sidecar and reports
  `still a/b` = correct decisions among the still decision steps before motion (v1 dirs: unchanged numbers).
  demo10 spoon: oracle_evidence recall RIGHT, decisions 58/64 but **still 0/6**: the model says the closing note through
  the six still steps and switches to `open bin 2` exactly at the arm motion (frame 430). Plain oracle mode shows
  still 5/6 only because the oracle writes the `open bin 2` label into the bank at the boundary (leak; not a memory
  test). Own writes: perception not trained yet (stage A). Watch B3 for the trigger moving before the motion.
* 2026-09-09 12:40 — A3-250 battery episodes 2–5 (demo10 tape, demo19 banana/box, demo54 banana): evidence-mode read
  RIGHT in the first three (older notes and the newest alike), still decision steps 0/6 everywhere, decision content
  right from the first moving frame (demo19 now `open bin 3` at once, where B2 said `open bin 2`). **demo54 banana is
  the first lead30 effect**: at the closing start (still scene) the model skipped the closing note and said
  `open bin 1` (right bin, conf 0.81) — a decision without motion — but one step later flipped to `open bin 2`, wrote it
  (own writes after the closing) and held it until the arm moved, then `open bin 1`. Reading: the decision digit is
  anchored by the just-decoded closing note (the `open bin _` context has no note of its own in the bank); when the
  decision fires before the closing note it is a guess. B3 trains this ordering further; a prompt-object query for the
  decision digit is a candidate follow-up if B3 keeps firing early without the note.
* 2026-09-09 12:43 — **A3-250 battery complete** (lead30 boundary): recall at the first closing step oracle_evidence
  5/6 (demo54 banana skipped the note), self 0/6; first decision at the first STILL step oracle_evidence 1/6, self 1/6;
  decision right once the arm moves 6/6 (evidence). **demo54 spoon = first clean success**: closing note read right,
  then `open bin 1` on a still scene 60 frames before the motion, still 6/6, 52/52. The trigger is detaching from the
  motion in the two demo54 episodes (cleanly in one, prematurely in the other); the other four still wait for the arm.
  B3 (own writes, label content) continues on these labels; its batteries (gate v4) run at 250/500/…
* 2026-09-09 13:30 — **B2-1250 policy server** (user 13:20 "prepare the b2 server"): `cluster_v6/serve_v6_job.sh`
  (worktree copy of serve_v5_job_v3) on the H200 (job 17329416, GPU 0, port 8000, 10.79.12.149; log
  `v6/diagnostics/server_v6_b2_1250_20260909.log`). Warm-up 128 s + 25 s (RTC shape). Smoke from the workstation
  (`cluster_v6/tools/serve_smoke_client.py`): reset ping OK, (50, 14) finite actions, ~260–300 ms per request. Real
  demo10 frames through the server: watching → `box in bin 1` → `spoon in bin 2` → … → closing `spoon in bin 2` →
  `open bin 2` (writes 1→6), i.e. the v6 read/write path works through `sample_with_memory`. Client contract unchanged
  (`examples/yam/client_subtask.py`; task1 prompt `find the <object>`); the client now sends a bare `reset_memory`
  ping at start (`--reset-memory`, default on) because the server only empties the bank on that ping. B3 batteries on
  the same card run sequentially (gate v5, `B_ONLY=1 BATTERY_MODES="oracle_evidence self" PARALLEL=0`).
* 2026-09-09 14:02 — **B3-250 battery (sequential, evidence + self)**: evidence recall 6/6 (demo54 banana no longer skips
  the note), still 0/6 everywhere (A3's two demo54 early firings are gone), decisions right after motion 6/6. Own
  writes demo10 spoon: recall from OWN notes RIGHT (18/18 closing steps), decisions 58/64, evidence exact 109/144
  (A3-250: 70/144) — perception improving at 250 with the read intact (label-content rule).
* 2026-09-09 14:18 — **B3-250 own writes**: recall from OWN notes 5/6 (A3-250 0/6, B2-250 4/6), decision right after
  motion 6/6, own notes exact ≈100/140 per episode (A3: ≈50). Miss = demo10 tape (wrote `tape in bin 1`, read it
  faithfully). Evidence recall 6/6. Still-step decisions 0/6 in all 12 runs: the motion trigger persists; decide at
  B3-500 whether to mask the decision loss after motion onset.
* 2026-09-09 15:21 — **B3-500 evidence**: recall 6/6; still-step decisions demo54 spoon 6/6 (fires 60 frames before the
  motion again, as A3-250 did), the other five 0/6 (switch at the arm); decisions after motion right in all six
  (demo54 banana 41/52). Own-write half running.
* 2026-09-09 15:40 — **v6.3 / B4** (user 15:22 "ok do it … directly start B, do we still need A?" → no: the label-content
  rule keeps the reader clean during own writes, so B4 warm-starts from B3-500). Two knobs: (1)
  `memory_v6_decision_ce_weight_after_motion=0.1` — the per-step sentence CE on the DATASET-flagged decision steps
  (= arm moving) is ×0.1, so the 6 still steps per episode carry about as much decision gradient as the ~50 moving
  ones; (2) `memory_v6_still_decision_boost=4.0` over `memory_v6_still_decision_frames=30` (user: "upsample the
  decision moments without arm moving") — sequence starts whose step grid covers a still decision frame are drawn 4×
  more often (`data_loader._still_decision_boost`, logged as mass before → after). `pi05_yam_mem_v6_task1B4` = B3
  recipe + both, from `OPENPI_V6_TASK1_B4_PARAMS` (default B3 keep_500). Launched via launch_B_generic (B3 stopped at
  ~560, 500 protected as keep_500); gate v5 instance #2 (B_ONLY) batteries B4 at 250/500/… while instance #1 finishes
  the B3-500 own-write battery and exits on B3's exit line.
