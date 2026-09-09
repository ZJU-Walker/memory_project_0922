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
