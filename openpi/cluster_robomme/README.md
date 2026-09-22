# RoboMME on the v7 boba memory recipe (worktree `memory_project_robomme`, 2026-09-13)

## 0. Recovery / layout

* Branch `robomme` of the memory_project repository (worktree of `/iris/u/kewalk/memory_project_v4/.git`), created
  2026-09-13 18:28 from v7 commit `ab09cac` (the boba two-phase-labels run; the repo the boba labelling session named).
  Tracked content only; read-only links `data -> memory_project_v4/data`, `v5 -> memory_project_v5/v5`,
  `v6 -> memory_project_v6/v6` (unused by the RoboMME configs, kept so every v7 config still resolves).
* Caches are REAL directories (`configure_v35_runtime_environment` rejects cache symlinks that leave the tree):
  `v35/cache/openpi/openpi-assets` = a 12 GB copy of `pi05_base` (from memory_project_v6), `v35/cache/openpi/big_vision`
  = the PaliGemma tokenizer, `v35/cache/huggingface/{hub,modules}` copies, `v35/cache/uv` empty, `v35/cache/jax` this
  tree's own compilation cache. `openpi/.venv` from `uv sync --frozen` (GIT_LFS_SKIP_SMUDGE=1, log `robomme/logs/uv_sync.log`).
* Every RoboMME artefact lives under `robomme/`: `data/lerobot/PickXtimes_v2` (dataset), `metadata/PickXtimes/`
  (manifest, official + v2 sidecars, prepared descriptors), `assets/PickXtimes/norm_stats.json`, `checkpoints/`,
  `logs/`, `diagnostics/`, `rollouts/`. Legacy copies `memory_v6_robomme` and `memory_v7_robomme_detail` are marked
  with a `LEGACY.md` and are reference only (their 161 GB of checkpoints were not deleted).
* Benchmark: `/iris/u/kewalk/robomme_benchmark` (its own `.venv`; datasets `data/robomme_data_h5`, 16 tasks x 100 demos).

## 1. What changed against v7 (user 18:17 / 18:22)

| knob | v7 boba2B | RoboMME PickXtimes | why |
|---|---|---|---|
| RTC | `simulated_delay=15` | `None` | the benchmark rollout is synchronous |
| memory step | 15 frames | **10 frames** | user |
| window / TBPTT / buckets | 60 / 30 / (20,40,60) | **40 / 20 / (20,30,40)** | episodes are 27-103 steps at stride 10; a pick-and-place cycle <= 27 steps |
| prefill | on, max 26 | **on, max 13** | analytic label-history prefill keeps windows short (user: "enable prefill to save gpu"); 12 segments max per episode + 1 |
| critical start pad | 75 frames | 50 frames | 5 steps at the new stride |
| action alignment | as recorded | **observation row t -> actions from t+1** (`action_target_offset_frames=1`) | the benchmark recorder stores the post-step observation on the action's row (`RecordWrapper.step`); the gripper width is already closing on the row of the first close command |
| cameras / state | 3 cameras, 14-D | front + wrist, right wrist zero + mask false; 8-D absolute joints (+ finger width), gripper +-1, no delta actions | `RobommeInputs` / `LeRobotRobommeDataConfig` |
| labels | boba v2 two-phase | **PickXtimes v2 two-phase** (§2) | user: "each pick has 2 subtasks" |
| warm start | boba pi05+KI base 9999 | **`pi05_robomme_PickXtimes_base_ki`** (§3) | there is no PickXtimes base 9999; the legacy base 7000 is plain pi05 without a subtask head |

Model code is untouched. Ported into the shared code (defaults keep every v7 config identical): `DataConfig.
action_target_offset_frames` (loader delta timestamps on both paths, `BuildMemorySequence` step mask, the sequence
sampler's per-frame budget), the loader's per-dataset camera keys, and `robomme_config.py` registration at the end of
`config.py`.

## 2. Two-phase labels v2 (`cluster_robomme/build_pickxtimes_v2_labels.py`)

Official `simple_subgoal` (planner view) already cuts each cycle into `pick up the <color> cube for the <k-th> time`
(reach, grasp at ~frame 80, lift) and `place the <color> cube onto the target` (carry, release, retreat); only the pick
carries the count, so at the next pick onset the bank's LAST value was count-free. v2 keeps the official boundaries and
gives the place phase its pick's ordinal: `place the red cube onto the target for the second time`. Every transition is
now the boba scoop/pour pattern: pick k -> place k (same count, loud grasp+lift) and place k -> pick k+1 (last + 1, loud
release+retreat); `press the button to stop` needs count == the repeat count in the prompt. 32 sentences (3 colours x 5
ordinals x 2 phases + press + `all tasks completed`), all lowercase; 4-12 segments per episode. Decision set = every
k >= 2 sentence + the button press (25), evidence = all pick/place sentences (30).

Dataset `robomme/data/lerobot/PickXtimes_v2`: the legacy lossless conversion (pixel-exact H264, 100 episodes, 53,720
frames, videos copied) with the task column / `meta/tasks.jsonl` / `meta/episodes.jsonl` rewritten to the v2 vocabulary.
Sidecar `metadata/PickXtimes/subtasks_v2.json` (schema v1, official segments kept per episode), descriptor
`prepared_v2.json` (SHA256 pins re-checked at config import, reference tokens = `sp.encode(lower) + "\n"`, max 12 tokens).
Manifest: all 100 released demonstrations are `train` (the benchmark val/test split is the evaluation).

## 3. Configs (`src/openpi/training/robomme_config.py`) and the plan

1. **`pi05_robomme_PickXtimes_base_ki`** -- non-memory pi05 + knowledge insulation on the v2 sentences (the
   `pi05_yam_boba0911_base` recipe: subtask + FAST CE train the VLM, flow trains the expert), continued from the legacy
   plain checkpoint 7000 (`robomme/checkpoints/pi05_robomme_base_PickXtimes/legacy_plain_r1/7000/params`, copied), no
   RTC, +1 offset, batch 16 on 2 GPUs, 5000 steps (warmup 200, 2.5e-5 -> 2.5e-6), EMA 0.999, checkpoints every 1000,
   final 4999. The count words are invisible in a single frame (every later pick starts from the target), so this base
   learns phase + colour and stays at chance on the ordinal -- exactly what the bank must supply.
2. **`pi05_robomme_mem_PickXtimes_B`** -- boba2B on that base: own write timing + label content, fresh memory leaves
   (`_V7_MEMORY_LEAF`), no stage A, lr 5e-5 flat after 100 warmup, batch 4 = two 40-step windows per GPU on 2 GPUs,
   2000 updates, checkpoints 500/1000/1500/2000. Loader path via `OPENPI_ROBOMME_BASE_PARAMS`.
3. `pi05_robomme_mem_PickXtimes_B_plain7000` -- the literal "skip everything" variant straight from the plain base (its
   sentence head starts untrained; own write timing is noise until the CE trains it). `pi05_robomme_mem_PickXtimes_A` --
   oracle-write fallback. Neither is the plan.

Launch (ON the job's node; `--gres` = all GPUs of the job, pinned with `CUDA_VISIBLE_DEVICES`):
`JOB=<job> GRES=<n> VISIBLE=<i,j> nohup setsid bash cluster_robomme/chain_pickxtimes.sh > robomme/logs/chain_pickxtimes_r1.out 2>&1 &`
(`SKIP_BASE=1` reuses an existing base checkpoint). Smoke: `cluster_robomme/smoke_all_testjob.sh` (test job 17405067).

## 4. Evaluation

`cluster_robomme/eval/` = the legacy harness (benchmark `BenchmarkEnvBuilder`, val/test episodes, joint_angle actions,
HTTP between the simulator venv and this tree's venv, annotated MP4s + traces under `robomme/rollouts/`):
`eval/serve.sh --task PickXtimes --stage B --checkpoint <robomme/checkpoints/pi05_robomme_mem_PickXtimes_B/<exp>/<step>>`
on a GPU, then `eval/run.sh --task PickXtimes --split val --episodes 0,1,2`. The execution chunk must equal the memory
stride (10 frames). Dev sanity before the simulator: `scripts/v5_heldout_video.py --config-name pi05_robomme_mem_PickXtimes_B
--params <ckpt>/params --episode-index <i> --write-mode {self,oracle} --manifest robomme/metadata/PickXtimes/manifest.json
--sidecar robomme/metadata/PickXtimes/subtasks_v2.json` (training episodes; there is no held-out demo split).

## 5. Status log

* 18:28 worktree, caches, dataset v2, base 7000 copy, venv (jax 0.5.3) -- done 18:38.
* 18:45 configs register (fields verified: stride 10, 40/20 steps, prefill 13, RTC off, offset 1, 32 sentences).
* 18:50 the CPU loader smoke segfaults on iris-ws-18 (exit 139, not OOM); on iris-hgx-1 inside the test job it PASSES
  for both configs (`robomme/logs/smoke_loader_node.log`): memory items = 40 steps x (2 cameras, 8-D state -> 32,
  50 x 32 actions), right wrist masked, per-step context 62-64 of 96 tokens, window at frame 250 of episode 0 -> 28
  valid steps (offset-aware) with 3 prefill sentences (pick 1, place 1, pick 2), 24 decision steps; base items 99-172
  of 224 tokens. Sampling: 100 full starts (5 %), 16,792 slice starts (45 %), 26,928 transition-anchored starts (50 %),
  buckets T20 24 % / T30 23 % / T40 53 %.
* 18:58 step-0 GPU smokes on the SHARED test H100 (13 GB used by other processes, our cap 48 GB): the memory config
  loads the plain base through the audited loader (51 arrays matched, 107 memory leaves fresh) and compiles, but the
  train step of ONE 40-step window at fsdp 1 needs 72.5 GB peak (XLA: "can't reduce below 16.8 GiB ... only reduced to
  72.51 GiB") -> OOM at the cap; the KI base at fsdp 1 has 53.6 GB of train-state arguments alone -> OOM. Neither is a
  code failure: both need the 2-GPU FSDP layout they are configured for (2 x H200: batch 4 = two windows per card is
  ~2/3 of the boba2B footprint that ran at 135 GB; 2 x H100 would need batch 2). No full optimizer step has run yet.
  Throwaway smoke checkpoint dirs removed; the test GPU was left as found (keep-alive + other users' processes).
* 09-15 14:20 (robomme_memory session) **base checkpoint check**: `legacy_plain_r1/7000` is byte-identical to the legacy
  `pickxtimes_base_full_r1_selected/7000` (17 files, 12,441,304,221 B, same `_METADATA`/`manifest.ocdbt`); plain pi05
  (51 param leaves: PaliGemma + action/time projections, NO subtask head, NO memory), trained from `gs://.../pi05_base`
  at batch 64 / FSDP 2 / lr 2.5e-5 cosine-15k on job 17356154 (2xH100), stopped at update 7180 of an 8000 segment,
  with the legacy offset 0 (the +1 alignment came later). The user accepted 7000 on 09-10 19:10 by explicit choice
  ("the current baseline is ok"), NOT a measured gate: the rollout gates at 500/1000/2000/4000 completed a first
  pick+place in 0-2 of 5 val episodes and never a whole episode; at 7000 only the action-chunk diagnostic on val ep 0
  exists (chunk 25: first pickup at step 136, placement at 218; chunks 5/10/50 never picked). Legacy memory line on
  top of it (reference only): selfwrite B1050 5/10 on the balanced val subset (all 1- and 2-pick, 0/2 at 3 and 4
  picks, 1/2 at 5), shortctx B2000 0/15 two-pick. So the base can pick and place but is not a benchmark-passing
  policy; the KI stage still has to teach the sentence head from scratch.
* 09-15 14:40 **boba5B port** (the boba session's current best: the scoop count moved from the cup to the bank only
  after blinding the count row; clips 29/33 vs 0-8, robot 23 runs count never wrong): git-applied memory_project_v7
  `e697f52` (pi0.py `digit_blind_rows` / `apply_digit_blind` / `_v7_digit_blind` in `_compute_sequence_loss_v32` and
  `decode_step`; pi0_config.py `memory_v7_digit_blind*` + `digit_blind_patterns()`; v5_heldout_video.py decode mask)
  onto this tree, uncommitted like the 09-13 work. New config **`pi05_robomme_mem_PickXtimes_B_blind`** = `_B` +
  `memory_v7_digit_blind=True` with explicit patterns `(28660, 604, 573)` = "cube for the" (pick) and
  `(4408, 604, 573)` = "target for the" (place): the row holding "the" predicts the ordinal token ("second"), which is
  a single PaliGemma token; the auto-derivation from digits does not apply (ordinal words, and the repeat count sits
  in the prompt, not in an "of n"). CPU check: the flagged rows are exactly the 30 ordinal-predicting positions of the
  32 reference sentences. `chain_pickxtimes.sh` takes `B_CONFIG` (default `_B_blind`; `_B` = unblinded control).
  Not ported (boba-only or later): the empty-scoop clip data, the `--stop-rule` phase ban at serving (PickXtimes'
  analogue would ban "pick" once the bank holds "place ... for the <prompt count> time"), the copy-bonus diagnostics.
* 09-15 14:45 **GPUs**: job 17403682 (4xH200, hgx-2) now carries the user's own Qwen3-VL robomme trainings on all
  four cards (joint_60k on 0,1 since 03:17; train_pickxtimes_ee_stage on 2,3 since 10:06) -> not available. The
  2xH100 job 17422715 (hgx-1, to 09-18 21:11) and the 1xH100 job 17434227 (hgx-1, to 09-17 18:32, same card as the
  old test job) hold only the 1 GB keep-alives; ownership not yet stated by the user. Test job 17405067 is gone.
* 09-15 15:00 **official labels are the plan** (user: "this time i still want to use the official provided subtask
  not our own labeled one"). `build_pickxtimes_v2_labels.py --label-version official` writes the released
  simple_subgoal sentences lowercased (20 sentences, only the pick carries the ordinal, same boundaries and frames) as
  `metadata/PickXtimes/subtasks_official.json` + `prepared_official.json` and the dataset
  `robomme/data/lerobot/PickXtimes_official` (3.6 GB, videos copied). `robomme_config.py` now registers both label
  versions: official configs carry the `_off` infix (`pi05_robomme_PickXtimes_base_ki_off`,
  `pi05_robomme_mem_PickXtimes_off_{B,B_blind,B_plain7000,A}`), the v2 names are unchanged. Decision set (official) =
  the 12 pick sentences with ordinal >= second + the button press; evidence = 18 pick/place sentences.
  `chain_pickxtimes.sh` defaults: BASE_CONFIG=`..._base_ki_off`, B_CONFIG=`..._off_B` (unblinded).
  Consequence of the official labels: before every pick onset the newest bank entry is the count-free place sentence,
  so the ordinal read must reach the last PICK (two writes back); the token-level keys of the v6 bank are built for
  that (the place sentence contains no "cube for the" context), but the boba "last + 1" shortcut does not apply.
* 09-15 15:10 **blinding, general form** (user: digit blinding "is too task specific"). `cluster_robomme/
  derive_memory_rows.py` derives from the label sidecar + prompts alone which sentence tokens NEED the memory: a trie
  node is MEMORY if prompt + prefix do not fix the next token but prompt + prefix + bank history always do; TIMING if
  even the history leaves >1 option; otherwise text/vision. PickXtimes official: 3 MEMORY rows (the ordinal after
  "pick up the <colour> cube for the"), 1 TIMING row (the first word: all/pick/place/press); v2: 6 MEMORY rows (pick +
  place ordinals), same TIMING row. So the hand-written boba5B patterns coincide with the label-derived rule; the
  rule needs no token surgery and transfers to any vocabulary. `_off_B_blind` uses `(28660, 604, 573)` = "cube for
  the" (15 flagged rows over the 20 sentences, CPU-checked); `_off_B` is the unblinded plan.
* 09-15 15:33 **single stage, self-write, from the plain base = the standard recipe** (user: "direct start stage B
  and use this for all later training, so no A/B any more, directly do self write ... should start from pi05 base on
  pickxtimes ckpt"). The KI base r1 (wandb 75f2htti) was stopped before its first update; the chain launcher is no
  longer the plan. New launcher `cluster_robomme/selfwrite_pickxtimes.sh` (CONFIG default
  `pi05_robomme_mem_PickXtimes_off_B_plain7000`, EXP default `pickxtimes_off_selfwrite_r1`, BATCH 2 on 2xH100).
  **Run r1 LIVE 15:34** on job 17422715 GPUs 0,1 (WANDB=1, project robomme_memory): official labels, own write
  timing + label content from update 0, fresh memory leaves, sentence head trained from scratch inside the stage,
  2001 updates, checkpoints every 500. Logs `robomme/logs/selfwrite_r1.out`, `train_pickxtimes_off_selfwrite_r1{,_status}.log`.
* 09-15 17:10 **eval prep for selfwrite r1** (run at update ~330, ce 13.7 -> 5.1 -> 2.9 -> 2.2 at 0/100/200/300,
  memory_grad_norm 2083 -> 4, ~12-16 s/update, wandb uej4l7g3). `run_mem_evals.sh` now requires JOB, picks the
  sidecar by config (`_off_` -> official) and takes SIDECAR/TAGSUF overrides. Counterfactual = oracle pass with
  `metadata/PickXtimes/subtasks_official_shift1.json` (`build_shifted_sidecar.py`: every pick ordinal +1 in the 87
  episodes with <= 4 picks; five-pick episodes unchanged): a bank-plus-one model answers the shifted ordinal (and
  presses the button one cycle early), a copier or a bank-ignorer answers the true one; together with the unshifted
  oracle pass (copier = off by one) this classifies the read. Scoring: `records[].pred` vs `gt_target` in the
  per-episode json, against both sidecars. Watch gotchas: tail -F needs `---disable-inotify` on NFS and every pipe
  stage after grep must be unbuffered (`sed -u`, `stdbuf -oL`). Eval GPU: none free in 17422715 (both cards hold the
  training); 17434227 (1xH100, keep-alive only) needs the user's word.
* 09-15 17:40 **eval GPU = our own A40 job** (user: "use a40 to eval, you can submit slurm yourself using iris").
  `cluster_robomme/eval_a40.sbatch` -> job **17452297** on partition iris (gpu:a40:1, 6 CPUs, 28 GB to fit iris5's
  leftovers, 2 days; the batch step runs the 1 GB keep-alive, evals are `--overlap` steps:
  `JOB=17452297 GPU=0 cluster_robomme/run_mem_evals.sh <config> <exp> <step>`). `run_mem_evals.sh` uses its own JAX
  cache `v35/cache/jax_eval_<gpu name>` so it never shares the training run's cache.
* 09-15 19:25 **checkpoint 500 of selfwrite r1** (H100 of 17434227 after the user's Qwen eval; eps 0/1/3/7 = 2/3/4/5
  picks; decision steps exact): self 14/21, 18/26, 19/35, 10/35; oracle 14/21, 24/26, 30/35, 30/35. Oracle traces:
  the ordinal chain second..fifth is right in every episode (one-step lag at each transition); misses = transition
  lags, the "all tasks completed" tail (garbage `Action: <loc..>` decodes) and the button press (missed in eps 0 and
  3, found in 1 and 7). Self-write failure = FLICKER: 1-2-step relapses to "pick ... second" inside the place phase
  get committed (12-18 writes for 6-12 segments); after such a write the model's previous own sentence is a pick, so
  the next pick onset is treated as the same pick (no +1) -> stalls at "second". `self_stable` is the WRONG tool here
  (its no-duplicate rule blocks the repeated place sentence: 3-4 writes, same scores); new eval mode `self_debounce`
  (debounce only, duplicates allowed) added to v5_heldout_video.py.
  **Counterfactual (proper, label version shift1 = dataset + sidecar with every pick ordinal +1, config
  `pi05_robomme_mem_PickXtimes_shift1_B_plain7000` on the _off params)**: at EVERY pick onset after the first the
  model outputs the shifted ordinal (eps 0/1/3: third at the true second pick, fourth at the true third, fifth at the
  true fourth; 7/7 onsets), the first pick stays "first" (empty bank). => BANK COUNTER (read + 1), not a picture
  reader and not a copier, already at update 500 with official labels and no blinding. The press fired in the shifted
  eps 1 and 3 (bank count > prompt count) but is missed when count == prompt in the unshifted eps 0 and 3: the stop
  comparison is the immature part. Gotcha: the eval's `--sidecar` only changes SCORING; oracle write tokens come
  from the loader (dataset task column + the config's pinned sidecar) -> a counterfactual needs its own label
  version/config (`videos_*_shift1_INVALID_scoring_only` = the first, invalid attempt). Renaming an output dir
  under a running battery loses its logs; never do that.
* 09-15 20:55 **why self-write stalls at "second" (ckpt 500 + 1000 traces, onset analysis)**: at the FIRST step of
  every pick k >= 3 the model outputs the ordinal of the last pick in the bank (copy), never k; only first -> second
  is learned (a "bank non-empty -> second" shortcut; 77/52/25/13 pick segments for second..fifth). The oracle's
  30/35 is copying: the oracle writes the true label at the onset step and the model copies it from the next step
  (the "one-step lag"). So the lying-bank test proved the bank is the source, but by COPY, not read + 1 (correction
  of the 19:25 note). Cause = the v6.2 label-content rule (`memory_v5_own_commit_label_content=True`, from task1 B2
  on 09-09): own timing commits the TRUE label at the first changed step, so "+1" is supervised for one step per
  pick and copying for ~14; at inference the model commits its own wrong "second", which locks the count. The
  codebase's A4 write delay (`memory_v5_write_delay_steps=1`) was the earlier remedy for the same "say the newest
  bank entry" failure; boba runs delay 0. Legacy robomme's full-own-content selfwrite (B250 -> 1050) was the best
  legacy result (5/10) but did not count past two either. ckpt 1000 self: 10/21 14/26 12/35 10/35 (no better).
  **Cost check:** the training scan already decodes the own sentence for the write timing; own content only swaps
  the tokens that enter the bank (pi0.py v5_bank_sentence) -> no extra compute, no caching needed.
* 09-15 20:56 **r2 = FULL SELF-WRITE** (user 20:51: "enable full selfwrite, still use prefill, start from our
  existing 1k ckpt"): config `pi05_robomme_mem_PickXtimes_off_B_own` = r1 config with
  `memory_v5_own_commit_label_content=False`, all params from r1/1000 (`OPENPI_ROBOMME_OWN_INIT` overrides), prefill
  on, 2001 updates, save 500. r1 stopped at update ~1270 (ckpts 500/1000 kept). Launch: `launch_own_r2.sh` ->
  `selfwrite_pickxtimes.sh` with CONFIG/EXP=pickxtimes_off_own_r2 on 17422715. The ckpt-1000 oracle / shifted /
  debounce-3 passes of r1 continue on the A40 (17452297). Eval videos: 256-px frames are now upscaled 3x with
  wrapped captions (`rerender_video.py`, eval renderer patched); viewer artifact = every run, two panels, sync play.
* 09-15 23:40 **r2/500 verdict + r3 prepared.** r2 (own content, from r1/1000) at 500: self 13/21, 15/26, 17/35, 15/35
  (r1: 14, 18, 19, 10); oracle 21/26, 23/35, 22/35 (r1: 24, 30, 30) = the model trusts the newest entry less.
  Onsets: third pick still COPY in every run (self, oracle, shifted); fourth pick no longer a copy but wrong ("fifth"
  from a bank ending in third); fifth pick a true +1 in ep 7. ce 0.61 -> 0.91 (100) -> 0.66 (600), still falling.
  Reading: own content gives the strongest +1 signal (after a wrong self-commit every remaining step of the pick says
  bank=k, target=k+1) but must beat a copy prior trained for 1000 updates plus the pointer copy bonus (beta ~10);
  second->third is the prior's stronghold. Decision: let r2 reach 1000 (battery queued on H100 self+shift, A40
  oracle). If the third-pick onset still copies at 1000 -> **r3** = `pi05_robomme_mem_PickXtimes_off_B_own_delay`:
  own content + `memory_v5_write_delay_steps=1` (A4: the bank never holds the current pick until one step after the
  onset, training and eval) + `memory_v6_pointer_beta` re-initialised at 0 (loader reinit allowlist), init r2/1000
  (`OPENPI_ROBOMME_R3_INIT`). Scripts: `logs/launch_own_delay_r3.sh`, `logs/stop_r2.sh`. Viewer artifact now lists
  runs as "r1 label-content" / "r2 own-content" (49 videos, transcoded to crf 30 for the 64 MB/version limit).
* 09-15 23:51 **r3 launched in parallel** (user: "stop 0 1 training and use that one" on the 4xH200 job 17403682):
  `pickxtimes_off_own_delay_r3` = own content + one-step write delay + pointer beta reset, from **r1/1000** (same init
  as r2, so r2 vs r3 isolates the two additions), batch 4 on H200 GPUs 0,1 (the user's current36 run keeps 2,3).
  r2/500 oracle complete: 13/21 21/26 23/35 22/35. Batteries queued: r2/1000 (H100 self+shift, A40 oracle), then
  r3/500 on the same split. Scripts `logs/{stop_ee6d_hgx2,launch_r3_hgx2,launch_r3_500}.sh`.

* 09-16 01:15 **r2 and r3 extended to 5000 updates** (user 01:12: "for r2 and r3 both make them train until 5k after
  our designed steps"). `_B_own` and `_B_own_delay` now carry `steps=5001` (save/keep 500 unchanged; the LR schedule
  was already flat at 5e-5 with decay_steps 10k, so nothing else moves). The running processes keep their in-memory
  2001 and stop at 2000 as designed; `logs/extend_own_r2_5k.sh` (hgx-1) and `logs/extend_own_delay_r3_5k.sh`
  (hgx-2) wait for that exit and relaunch `selfwrite_pickxtimes.sh`, which resumes from ckpt 2000 (run_train.sh
  resume policy, W&B run resumed by id) and runs to 5000; at most 3 relaunches, logs `logs/extend_*_5k.log` and
  `logs/own_*_extend.out`. Expected: r2 ~10.8 s/update -> 5000 around 13:30; r3 ~18 s/update -> around 01:00 09-17
  (job 17403682 ends ~15:50 09-17). r2/1000 self: 13/21, 13/26, 14/35, 23/35; ep3's first decision over-counts
  ("third" at the second pick), ep7 counts second->third->fifth(flicker)->fifth.
* 09-16 01:36 **r2/1000 battery complete** (viewer Version 5, 68 videos). Decision steps correct, eps 0/1/3/7 (2/3/4/5
  picks): self r1/1000 10/21 15/26 12/35 10/35 -> r2/500 13 15 17 15 -> r2/1000 13 13 14 **23**; oracle r1 11 21 25 30
  -> r2/500 13 21 23 22 -> r2/1000 13 20 28 30; shifted bank r1 10/33 20/38 27/47 30/35 -> r2/500 0 14 20 22 -> r2/1000
  4 20 29 30 (ep7 has five picks so its shifted labels equal the official ones = control). Onset classes (scratchpad
  `onset.py`, `transitions.py`): oracle ep7 is +1 at every onset (second..fifth) for the first time in this project;
  oracle ep3 +1 at the third, copy at the fourth; oracle ep1 still copies at the third. Self: "second" is always
  written during the first place (a +1 from bank=first) in every run, so pick 2 was never the issue; r2/1000 ep7
  increments to "third" at the pick-3 onset, then jumps third->fifth during place 3 (the oracle pass flickers "fifth"
  at the same frame -> a visual trigger, not the bank), ep3 writes "second" and "third" in consecutive steps during
  place 1 and stays one ahead, ep1 reaches "third" only at s53 (late). The button press is never predicted in self
  mode (0-1 of 5-7 steps). Reading: own-content training moved the model from copy to +1 (the ce fell 0.61 -> 0.49
  at 1000); the residual errors are commit flicker (double +1, +2 skips) and the stop decision. r2 and r3 run on to
  5000; r3/500 battery auto-starts when its ckpt lands (~02:15; `h100_r3_step.sh` on hgx-1, `a40_r3_step.sh` on iris5).
* 09-16 02:25 **r3/500 battery moved entirely to the A40.** The H100 of 17434227 now runs the user's own
  PickXtimes_current36 challenge eval (policy `serve` 12 GB + `episode.py`); an eval step next to a live policy
  server would disturb its timing, so `h100_r3_step.sh` was stopped before it launched and `logs/a40_r3_rest.sh`
  (iris5) runs self + shifted after the A40 oracle pass (`a40_r3_step.sh`). Order on the A40: oracle 7 1 3 0, self
  7 1 3 0, shifted 0 1 3 7 (~2 h total). r3 ckpt 500 landed 02:22 (ce 0.51).
* 09-16 02:50 **r3's pointer-beta reset is unrecoverable -> r3 is not the experiment it was meant to be.** r3/500
  oracle ep7 = 10/35, first decision "fifth"; the trace says "third"/"fifth" at high confidence whatever the bank holds
  (bank=first -> "third"). Checkpoint scalars (`params.memory_v6_pointer_beta.value`, read via tensorstore/ocdbt):
  r1/500 10.0015, r1/1000 10.0088, r2/500 10.0088, r2/1000 10.0120, **r3/500 0.0066**. Adam at lr 5e-5 moves a
  scalar ~0.05 per 1000 updates, so the copy bonus that reads the ordinal out of the bank is gone for the whole run
  and the model falls back to a visual prior. The delay lever is therefore untested. Prepared: config
  `_B_own_delay_keepbeta` (delay only, beta kept, init r1/1000, 5001 steps), `logs/launch_r3b_hgx2.sh`,
  `logs/stop_r3.sh`. Confirmation queued on the A40 (`a40_r3_rest2.sh`): oracle eps 1 3 0, then oracle eps 7 3 with
  beta forced to 10 (`v5_heldout_video.py --set-param memory_v6_pointer_beta/value=10 --tag-suffix _beta10`, new
  flags), then self 7 1 3 0; the r3 shifted pass is dropped.
* 09-16 02:55 **r3 stopped at update ~640, r3b launched** (user 02:53 "Ok just run r3b"). r3/500 oracle: 10/35, 8/26,
  9/35 (first decisions fifth/third/fourth), the count detached from the bank in every episode. r3b =
  `pi05_robomme_mem_PickXtimes_off_B_own_delay_keepbeta/pickxtimes_off_own_delay_r3b`: own content + one-step write
  delay, pointer beta kept (~10), init r1/1000, 5001 updates straight through (no extend waiter needed), batch 4 on
  H200 GPUs 0,1 of 17403682. r3's ckpt 500 stays for the beta-10 counterfactual + self pass still queued on the A40.
* 09-16 03:10 **beta-10 counterfactual confirms the diagnosis.** r3/500 oracle ep7 with `memory_v6_pointer_beta`
  forced back to 10 at load time: 10/35 -> 24/35, first decision "second" (OK), and the ordinal follows the bank
  again (every onset = the bank's newest pick, i.e. r1-style copying: k3/k4/k5 copy). So the reset alone killed the
  bank read; whether the write delay helps is left to r3b. r3b step 0 ce 0.57 (r3 after the reset: 0.86 at 100).
* 09-16 03:35 **Training loss is blind to the copy bonus.** r3 (beta 0) and r3b (beta 10) log the same ce at step 0
  (0.5732 vs 0.5747) and step 100 (0.8613 vs 0.8617), same data order, while the eval swings 10/35 -> 24/35. The
  sequence loss does add `v6_pointer_bonus` (pi0.py `_compute_sequence_loss_v32`), and there is NO held-out split
  (all 100 demos train; eval eps 0/1/3/7 are training episodes), so the gap is train-mode vs free-running on the SAME
  episodes: under teacher forcing the ordinal token is predictable without the pointer (label prefix tokens, label
  prefill, memorised visuals), while the free-running decode leans on the x10 pointer bonus, which pushes the retrieved
  (newest) ordinal = the copy behaviour. The +1 rule therefore gets little gradient through the pointer. Candidate
  levers: beta fixed at ~3 (not 0), or a training objective that runs the decode free (own tokens as context).
* 09-16 03:55 **r2/1500 ep7 self = 7/35: back to pure copying** ("second" from the first place to the end; r2/1000 had
  the correct third and 23/35). The +1 behaviour at 1000 was not a stable state of the run; own-content training keeps
  oscillating between copy and +1 on this episode. r3/500 self complete: 10/35 8/26 9/35 12/21 (detached count).
* 09-16 04:20 **Why own-content training still copies (mechanism).** Training decision accuracy is 88 %
  (`v5_exact_decision_sum/v4_decision_count` at r2/1000) vs 40-65 % free-running on the SAME episodes. The history
  prefill (`transforms.py _history_prefill`, delay 0) hands every window the correct label history, and inside a
  window every non-onset row is answered correctly by copying the newest bank entry (~13 of 14 rows per pick).
  Own-content writes only create "+1" supervision when the model's own commit was wrong, which under a correct prefill
  is rare; free-running rollouts accumulate exactly the errors training never shows. The copy bonus (beta 10) then
  makes copying the free-running default. `memory_critical_prob=0.5` (starts <=5 steps before a transition) puts an
  onset near many window starts but keeps the 1:13 ratio. No onset/transition weight exists in the loss. Candidate
  fix for the morning: an onset-weighted sentence CE (rows whose label differs from the previous step's label, or from
  the last prefilled sentence at t=0, weighted ~x10), plus beta ~3; r3b's delay adds one +1 row per segment.
* 09-16 04:40 **Onset-weighted sentence CE implemented (default off).** `Pi0Config.memory_v7_onset_ce_weight` (1.0 =
  unchanged); pi0.py `_compute_sequence_loss_v32` builds `xs["onset_mask"]` = steps whose label sentence span
  (`tokenized_causal[:, :, :memory_v5_sentence_len]`) differs from the previous step's (step 0 of a window never an
  onset) and multiplies the per-step CE by the weight there, after the v6.3 decision weight. Config
  `pi05_robomme_mem_PickXtimes_off_B_own_onset10` (r4 candidate: own content, beta kept, onset x10, init r1/1000,
  5001 updates; `OPENPI_ROBOMME_R4_INIT` overrides) imports; the running configs report onset 1.0, so nothing
  running or queued changes. Launcher prepared: `logs/launch_r4_onset.sh` (needs a free GPU pair; JOB/VISIBLE env).
* 09-16 04:30 **Pointer gain 3 at eval (r2/1000 self, ep7): 23/35 -> 9/35.** The bank still says "second", but at
  s22 (the frame where gain-10 self and the oracle pass also flicker "fifth") the visual guess wins with the weaker
  bonus, "fifth" is committed and copied to the end. So the x10 bonus is not only the copy prior, it is also what
  suppresses the visual over-count; the base logits alone do not carry a count. Sweep continues: ep3 at 3, then 0, 1.
* 09-16 05:00 **Pointer-gain sweep at eval (r2/1000 self, eps 7 / 3):** gain 10 = 23/35 / 14/35, gain 3 = 9/35 /
  13/35, gain 0 = 11/35 / 8/35, gain 1 = 9/35 (ep7). Every reduced-gain run commits "fifth" (ep7) or "fourth"/"fifth"
  (ep3) at the first decision: without the bonus the base logits carry no count, only a visual guess. The bonus is the
  copy prior AND the guard against visual over-counting -> weakening it at eval is not a lever; the training signal
  is (onset weight, r4). r2 exited at 2000 (ce 0.39) and was resumed to 5000 at 04:57 (mode=resume, W&B run kept).
* 09-16 05:22 **r2/2000 self: ep7 7/35, ep1 14/26 = pure "second" copying, same as 1500.** Own-content training has
  settled into the copy solution (ce keeps falling: 0.49 @1000, 0.46 @1500, 0.39 @2000, all under the label prefill);
  the correct third pick at 1000 was transient. r2 resumed to 5000 as instructed; recommendation to the user: give the
  H100 pair to r4 (onset-weighted CE) instead.
* 09-16 05:50 **r2/2000 oracle: ep7 28/35, ep3 29/35, third pick = copy in both** (the +1 onsets of r2/1000's ep7
  oracle are gone). r2/2000 self 13/21 14/26 18/35 7/35. Whole r2 line: copy solution at 500, +1 flicker at 1000,
  copy again at 1500 and 2000.
* 09-16 06:00 **r3b/500 oracle ep7 = 28/35 with +1 at the second, third AND fifth onsets** (r2/500 oracle: 22/35,
  third = copy; r2/2000 oracle: third = copy). The one-step write delay does what A4 promised: the onset row sees the
  bank without the current pick and answers "newest + 1". The fourth pick is still lost to the s51 "fifth" visual
  flicker (every run flickers there). Self-write pass pending.
* 09-16 06:30 **r3b/500 self ep7 = 16/35, 32 writes: the +1 rule is learned, the self-write timing breaks it.** Oracle
  eps: 13/21 21/26 25/35 28/35 with +1 at every second/third onset. Self ep7: "second" written during the first place
  (an early flicker at a place step that own timing commits; the labels are NOT shifted, subtask_lookahead=0), then at the pick-2 onset the row sees bank=second and applies +1
  again -> "third"; from there it alternates 2/3/2/3 (s24-28) and 3/4/3/4 (s37-44), one write per step, and runs one
  ahead (fourth at pick 3, fifth at pick 4). In training the pending/prev sentence tells the row whether the newest
  bank entry is its own announcement (copy) or the previous pick (+1); the self-mode eval seems to lose that
  distinction with delay 1 -> check the eval's pending handling vs the training scan before judging r3b.
* 09-16 06:33 **r3b/500 self ep1 = 21/26 = its oracle score: the first fully correct self-written count in this
  project** ("second" at s19 during place 1, held through pick 2, "third" at s40 during place 2, no oscillation, 12
  writes). r2 never left "second" on this episode (13-15/26 at every checkpoint).
* 09-16 06:40 **r3b/500 self ep3 = 6/35: one ahead from pick 2.** "second" at s19 (place 1), then the pick-2 onset
  row applies +1 again -> "third", fourth from place 2 on; no oscillation. So r3b/500 self = 16/35 (ep7, one ahead +
  oscillation), 21/26 (ep1, perfect), 6/35 (ep3, one ahead): the +1 rule is there, the "newest entry is my own early
  announcement -> copy" case is not yet reliable. r3b/1000 battery queued (self first).
* 09-16 06:45 **r3b/500 battery complete.** Oracle 13/21 21/26 25/35 28/35 (every second/third onset +1); self 13/21
  21/26 6/35 16/35 (ep0 and ep1 = their oracle scores, ep3/ep7 one count ahead after the anticipatory write). Next:
  r3b/1000 (self first) when the ckpt lands ~07:15. Button press still never predicted in self mode.
* 09-16 07:33 **r3b/1000 self ep7 = 5/35 (31 writes): the k/k+1 oscillation now dominates.** At the pick-2 onset the
  bank rule says "third" (+1 from the early "second"), vision says "second"; each output flips pending/bank and the
  next row answers the other way (s24-s31 2/3/2/3), the bank fills with alternating entries, and the model then locks
  on "second" for picks 3-4 (copy) before oscillating again at pick 5. With own-content writes + delay, the training
  windows also contain such wrong-bank rows and their labels teach "-1 from the bank" half the time -> the model learns
  to let vision veto the bank, which is exactly the alternation. Waiting for eps 1/3/0 before drawing the line.
* 09-16 07:42 **r3b/1000 self ep1 = 14/26 (was 21/26 at 500):** "second" copied through pick 3, "third" only at s53.
  r3b/1000 self so far: 5/35 (ep7), 14/26 (ep1) -> the delay's gains at 500 did not hold at 1000; own-content
  training keeps drifting toward copy + vision veto.
* 09-16 08:02 **r3b/1000 oracle ep7 = 26/35 with the third pick a COPY again** (500: +1). r3b/1000 self 13/21 14/26
  8/35 5/35. Pattern across r2 and r3b: a transient "+1" state early (r2 @1000, r3b @500) that decays into copying as
  own-content training continues; the 1:13 onset:copy supervision ratio wins in the end. r3b/1500 self queued as the
  trend check; the principled fix remains r4 (onset-weighted CE), awaiting the user's GPU decision.
* 09-16 08:10 **r3b/1000 battery complete** (viewer Version 8, 108 videos): self 13/21 14/26 8/35 5/35, oracle ep7
  26/35 (third = copy), ep3 28/35 (third = +1, fourth = copy). r3b/1500 self queued (`a40_r3b_1500.sh`).
* 09-16 09:45 **r3b/1500 self ep7 = 7/35, pure "second" copying** (500: 16/35 with +1 at pick 3; 1000: 5/35
  oscillating). r3b has converged to the same copy solution as r2; the delay alone cannot beat the 1:13 onset:copy
  supervision. Both own-content runs (r2 -> 5000 on the H100s, r3b -> 5000 on the H200s) are now low-value; the
  onset-weighted r4 needs one of the pairs. A second candidate for the other pair: onset x10 + delay 1
  (`_B_own_onset10_delay`, not yet registered).
* 09-16 10:00 **r3b/1500 self: 7/35 (ep7), 14/26 (ep1), 15/35 (ep3) = "second" copied through every later pick**
  (ep3 oscillates 2/3 at the very end, 23 writes). Converged to the r2 copy solution. r4 (`_B_own_onset10`) and r4b
  (`_B_own_onset10_delay`, registered 09:50, launcher `logs/launch_r4b_onset_delay.sh`) wait for the user's GPU call.
* 09-16 11:40 **Correction: the labels are not shifted** (`subtask_lookahead=0` in the robomme config). The early
  "pick k" during place k-1 is a flicker of the prediction at a place step (ep7 s22, ep0 s13, ep3 s19) that the
  own-timing write rule commits immediately; the real pick then starts 1-8 steps later and the row adds one again.
  Queued on the idle A40: r3b/500 `self_debounce` with 2 and 3 stable steps on eps 3 7 1 0 (`a40_r3b_500_debounce.sh`).
* 09-16 11:56 **Debounce (2 stable steps) on r3b/500 self: ep3 6/35 -> 17/35, ep7 16/35 -> 23/35.** The k/k+1
  oscillation never reaches the bank (consecutive predictions differ), the count is right through pick 3 in ep7
  ("third" at the onset, +1); what remains is the s51 visual "fifth" flicker (held 2 steps, so it commits) and ep3
  drifting one ahead at pick 3. Eps 1/0 and the 3-step variant pending. Gradual-write design (delta_rate <1, write
  every confident step, prefill at rate 1) described to the user 11:52, awaiting rate/delay/GPU choice.
* 09-16 12:05 **r2 and r3b stopped (user 11:57/11:58), r5 + r6 launched, both with onset x10 (user 12:03).**
  Code: `MemoryConfig.delta_rate` now overridable per write (`delta_write_kv[_multi](..., rate=)`); the label-history
  prefill writes at rate 1.0; `Pi0Config.memory_v7_write_every_step` (write on every confident step) and
  `memory_v7_write_debounce_steps` (commit only a sentence produced N steps in a row; new scan carries
  `last_cand_sentence`/`cand_streak`); the v4 pin `delta_rate==1` relaxed for the every-step mode; the eval applies
  the model's debounce in plain self mode and lists distinct runs when writes are every-step.
  - **r5** `pi05_robomme_mem_PickXtimes_off_B_own_delay_grad03/pickxtimes_off_own_delay_grad03_r5`: own content,
    delay 1, beta kept, bank rate 0.3 + every-step writes, onset x10, init r1/1000, 5001 updates, H200 GPUs 0,1 of
    17403682 (GPUs 2,3 = the user's Qwen run, untouched), batch 4. Launched 12:04.
  - **r6** `pi05_robomme_mem_PickXtimes_off_B_own_delay_deb2/pickxtimes_off_own_delay_deb2_r6`: own content, delay 1,
    beta kept, training debounce 2, onset x10, init r1/1000, 5001 updates, H100 GPUs 0,1 of 17422715, batch 2.
  r2 stopped at ~3900 (ckpts 500..3500 kept), r3b at ~2100 (500/1000/1500/2000 kept). Batteries queued on the A40
  (`a40_r5_r6_500.sh`, after the r3b/500 debounce test): self 7 1 3 0 + oracle 7 3 for each at 500.
  Debounce test so far (r3b/500, 2 steps): ep3 17/35, ep7 23/35, ep1 20/26.
* 09-16 12:20 **Onset-mask bug fixed before it trained anything.** The first r5/r6 launches logged ce 5.8/6.3 at step 0
  (r3b: 0.57): the onset mask compared the raw first sentence_len causal positions, which hold FAST action tokens
  after a short sentence, so nearly every step counted as an onset and got the x10. Now only sentence tokens (causal
  mask and not FAST) are compared; `v7_onset_count` is logged. Also (user 12:11): r5/r6 save every 200, keep the newest
  2 plus multiples of 600 (`TrainConfig.checkpoint_max_to_keep`, orbax max_to_keep now configurable), W&B on;
  deleted the never-evaluated checkpoints r2/2500-3500 and r3b/2000 (108 GB). A40 batteries re-queued at 600, 1200,
  1800, ... for both runs (`a40_r5_r6_steps.sh`).
* 09-16 12:33 **r3b/500 debounce check complete.** 2 stable steps: 13/21 20/26 17/35 23/35; 3 steps: 13/21 18/26 6/35
  14/35; plain: 13/21 21/26 6/35 16/35. Two steps is the setting (r6). r5 step 0 ce 1.21, r6 1.36 with the fixed
  onset mask (label changes = 11 % of steps, x10).
* 09-16 13:25 **Early look r6/200 self ep7 = 15/35, 11 writes:** "second" during place 1, "third" at the pick-3
  onset (+1), no oscillation, no double count, the s51 "fifth" flicker no longer commits; but the count stays at
  "third" through picks 4 and 5 (copy). 200 updates only. r5/200 and ep3 follow; 600-step batteries at ~15:00.
* 09-16 13:40 **Eval split (user "can you use the h100"): H100 of 17434227 (free again, keep-alive only) runs r5,
  the A40 runs r6.** `logs/h100_r5_evals.sh`: r5/200 self eps 3 1 0 now, then r5 at 600/1200/... (self 7 1 3 0, oracle
  7 3 up to 1200); `logs/a40_r6_evals.sh`: r6/200 self eps 1 0 after the running r5/200 ep7, then r6 at the same
  steps. Viewer: run + mode filters on the score table (Version 14). r6/200 self: ep7 15/35, ep3 19/35.
* 09-16 13:45 **r5/200 (gradual bank) self: ep3 = 30/35 (best on this episode by 11 steps; second -> third -> fourth
  each written during the preceding place, no double count, 80 every-step writes), ep7 = 7/35 (stuck at "second",
  89 writes).** r6/200 self: ep7 15/35, ep3 19/35. Viewer Version 15.
* 09-16 13:50 **r5/200 self complete: 13/21, 14/26, 30/35, 7/35** (eps 0 1 3 7; 52/61/80/89 every-step writes).
  Viewer Version 17. r6/200 eps 1 0 running on the A40; both runs' 600 checkpoints ~15:00 (r5 -> H100, r6 -> A40).
* 09-16 13:58 **Step-200 early look complete.** r5 (gradual bank): 13/21 14/26 30/35 7/35; r6 (training debounce):
  13/21 21/26 19/35 15/35. No oscillation or double count in any of the eight traces; r5 locks on "second" in eps 1/7
  (14-step reinforcement of the held sentence), r6 counts to third on eps 1/7 but not beyond. Next: 600 (~15:00).
* 09-16 14:23 **Step-400 look (eps 7 / 3):** r6 15/35 / 19/35 (identical traces to its 200), r5 7/35 / 23/35 (ep7 now
  counts one pick LATE: "third" during place 3, "fourth" during place 4; ep3 down from 30 at 200). Viewer Version 23.
* 09-16 15:00 **r6/600 self ep7 = 27/35 with ALL FIVE picks counted (+1 at every onset: second, third, fourth,
  fifth), 11 writes, no double count** - the first fully correct five-pick self-written count in this project (previous
  best 23/35 with the fourth lost). The s51-55 visual wobble ("fourth"/"fifth"/"fourth") never commits thanks to the
  training debounce. Button press still missed (1/5). r6 = own content + delay 1 + debounce 2 + onset CE x10, from
  r1/1000. Viewer Version 24.
* 09-16 15:18 **600-step batteries so far.** r6: ep7 27/35 (all five picks), ep1 8/26 (two-step early "second" passes
  the debounce -> double count), ep3 6/35 (same: "second" then "third" during place 1, two steps each). r5: ep7 8/35
  (one pick late + third/fourth alternation), ep1 21/26 (clean count). Viewer Version 27.
* 09-16 15:23 **600 self sets complete.** r6 (debounce): 12/21, 8/26, 6/35, 27/35; r5 (gradual): 13/21, 21/26,
  17/35, 8/35. Oracle 7/3 pending for both.
* 09-16 15:30 **r6 stopped by the user at ~760 (ckpts 400/600 kept); r7 launched on the H100 pair.** Early-run
  analysis of r6/600 (`scratchpad`, "early runs"): the writes that cause the double counts are LONG (9-13 steps)
  announcements of pick k during place k-1 or of k+1 during pick k, not flickers -> a longer debounce would not help.
  r7 = r6 recipe warm-started from r6/600 + `memory_v7_onset_ce_pre_steps=2` (the 2 steps before every label change
  also get the x10, penalising early announcements) = `pi05_robomme_mem_PickXtimes_off_B_own_delay_deb2_pre2/
  pickxtimes_off_own_delay_deb2_pre2_r7`, 5001 updates, save 200. A40 queue `a40_r7_evals.sh` (looks at 200/400,
  batteries at 600/1200/...). r5 continues on the H200s (evals on the H100 of 17434227). 600 oracle: r5 ep7 30/35 ep3
  25/35; r6 ep7 27/35.
* 09-16 15:37 **r7 stopped 1 min after launch (user: "dont start r7 yet, i need the h100 for other thing"); H100 pair
  free (keep-alives only), A40 queue cancelled. 600 oracle complete:** r6 ep7 27/35 ep3 27/35 (+1 at every onset),
  r5 ep7 30/35 ep3 25/35 (k4 copy). With a clean bank both count; the self-write gap is the write timing. Viewer
  Version 30 (144 videos). r5 keeps training on the H200s; its evals stay on the single H100 of 17434227.
* 09-16 16:20 **r5/r6 verdict: where the decision steps go, and why (analysis before the next run).** r6/600 self
  12/21 8/26 6/35 27/35 = 53/117 decision steps; of the 64 lost, 39 are the count one AHEAD after an early write (eps 1,
  3), 22 are the button press never said (every episode), 3 are one-step boundary flickers. r5/600 (gradual bank,
  51-89 writes per episode): 59/117, lost 21 press, 29 one BEHIND (ep7 holds "second" through pick 3, ep3 holds "third"
  through pick 4) and 7 ahead: every-step writes reinforce whatever is said, wrong as much as right; not the path.
  r3b/500: 56/117 (22 press, 37 ahead). Two mechanisms, both measured:
  - **Early write -> double count.** Listing every pick sentence produced inside a place phase (all self traces): the
    harmful ones (ordinal above the bank's newest pick, held 2+ steps, committed) all sit at step +1 of a place, i.e. the
    frame right after the release: the arm lifts and the model says "pick k+1" for two steps, which passes the debounce
    and is committed BEFORE "place" is; "place" then follows it in the bank, the bank reads like a whole extra cycle,
    and the true onset adds one more. Frames of ep0 (f120 release, f130-f210 retreat/hover, f210 = labelled pick-2
    onset) are indistinguishable: the label boundary is a planner boundary, not a visual event, so the x10 onset row
    trains "pick k+1" on late-place-looking frames; some early triggers are unavoidable and the write rule must survive
    them. A longer debounce cannot (3 steps was worse: places last 5-11 steps and "place" must be held that long too).
  - **Press never learned.** Teacher-forced probe `cluster_robomme/probe_press_tf.py` (the training loader's own window
    anchored 2 or 10 steps before the press onset, label-history prefill, `compute_loss(train=False)`, per-step
    exactness) on r6/600: press rows 0/7 1/6 1/6 1/5 while the pick rows of the same windows are 3/3 4/4 3/3 3/3. So
    the press fails under exactly the training conditions; the 92-97 % training decision exactness is the pick rows.
    Frames f420-f480: under the press label the arm retreats to the top of the image and touches the button only under
    "all tasks completed", so vision offers little; the decision has to be bank count == prompt count, and the ~630
    press rows per epoch (weight 1) never won against ~4000 pick/place rows whose answer the copy bonus supplies.
    r1/500 still pressed 4-5/6 (visual + bank) and lost it by r1/1000; every later run says "pick k" through the press.
  - Code (default off; every existing config imports unchanged, CPU-checked): `Pi0Config.memory_v7_write_grammar` +
    `memory_v7_write_grammar_initial` = label-derived PHASE GRAMMAR write gate (robomme_config `_phase_grammar` reads
    the sidecar: PickXtimes official = pick->place, place->pick, place->press, press->all, start pick; training scan
    `Pi0.v7_grammar_allows` on the first token of the candidate vs the newest COMMITTED sentence, rejected candidates
    retried, `v7_grammar_rejected_count` logged; the eval applies the same rule in self mode and records
    `grammar_rejections`). `memory_v7_kind_ce_weights` = per-phase sentence CE weight from the label's first token
    (takes the max with the onset weight, no stacking). Offline replay of the r6/600 traces under the gate rejects
    exactly the two harmful writes (ep1 s15, ep3 s15) and nothing in eps 0/7.
  - **r8 design** `pi05_robomme_mem_PickXtimes_off_B_own_delay_deb2_gram` (exp `pickxtimes_off_own_delay_deb2_gram_r8`):
    the r6 recipe (own content, delay 1, debounce 2, onset x10, beta kept, prefill) warm-started from r6/600 (the
    checkpoint that adds one at every onset given a clean bank) + the phase-grammar gate + press rows x10; 5001 updates,
    save 200, keep newest 2 + multiples of 600. Launchers `logs/launch_r8_hgx2.sh` (H200 0,1 of 17403682, batch 4,
    after `logs/stop_r5.sh`) and `logs/launch_r8_hgx1.sh` (H100 pair 17422715, batch 2); evals `logs/a40_r8_evals.sh`
    (200/400 self eps 1 3 7; 600/1200 self 7 1 3 0 + oracle 7 3). NOT launched: r5 and the GPUs are the user's call.
    Gate-only check of r6/600 (same params, r8 config at eval time) started on the H100 of 17434227 ->
    `videos_pickxtimes_off_own_delay_deb2_r6_600_gram` (eps 1 3 7 0). A40 queue: probe on r1/500, r6/600 oracle eps
    0 1, probe on r1/1000 and r2/1000.
* 09-16 17:00 **Both checks landed.** (1) Gate-only: r6/600 weights evaluated with the r8 config (phase-grammar write
  gate at eval time only, no retraining), self mode: ep1 8/26 -> 19/26 (2 notes refused), ep3 6/35 -> 26/35 (2 refused),
  ep7 27/35 -> 27/35 and ep0 12/21 -> 12/21 (0 refused) = 53/117 -> 84/117; every final bank now counts correctly
  (`videos_pickxtimes_off_own_delay_deb2_r6_600_gram`); the remaining losses are the press (22) and one-step flickers.
  (2) Press probe across checkpoints (press rows exact under training conditions, eps 0/1/3/7): r1/500 3/7 6/6 4/6 3/5,
  r1/1000 3/7 4/6 3/6 3/5, r2/1000 0/7 3/6 1/6 1/5, r6/600 0/7 1/6 1/6 1/5 -> partly learned early with label-content
  writes, then forgotten step by step under own-content training; the pick rows of the same windows stay 3/3-4/4
  throughout. r6/600 oracle eps 0/1: 12/21, 20/26 (press missed there too). Nothing launched (user: analysis first).

## Run track (from 09-17 the runs are named `robomme_MMDD_vN`; r1-r6 are history)

| run | config | init | recipe | GPUs | status |
|---|---|---|---|---|---|
| r1 | `_off_B_plain7000/pickxtimes_off_selfwrite_r1` | plain 7000, fresh memory | own timing, LABEL content, delay 0 | 2xH100 | copies at k>=3; ckpts 500/1000 |
| r2 | `_off_B_own/pickxtimes_off_own_r2` | r1/1000 | own content | 2xH100 | transient +1 at 1000, copy after; stopped ~3900 |
| r3 | `_off_B_own_delay/..._r3` | r1/1000 | + delay 1, beta RESET (broken) | 2xH200 | stopped 640 |
| r3b | `_off_B_own_delay_keepbeta/..._r3b` | r1/1000 | + delay 1 | 2xH200 | 500 good, decays; stopped ~2100 |
| r5 | `_off_B_own_delay_grad03/..._r5` | r1/1000 | + delay 1, every-step writes at rate 0.3, onset x10 | 2xH200 | mirror loop, count behind; stopped 747 by the user |
| r6 | `_off_B_own_delay_deb2/..._r6` | r1/1000 | + delay 1, debounce 2, onset x10 | 2xH100 | best: 27/35 five picks; double counts eps 1/3; stopped ~760 |
| **robomme_0916_v0** | `pi05_robomme_0916_v0/robomme_0916_v0` | **plain 7000, fresh memory** | own content, delay 1, debounce 2, prefill, linear delta bank, **retraction 3, vocabulary-only writes, per-token hard-word CE x10** (no onset weight) | 2xH100 17422715, batch 2 | launched 09-17 00:11 |

* 09-16 22:05 **user: no task-specific mechanisms.** The phase-grammar write gate and the press-row weight are rejected
  ("you are again making this task specific"); their knobs stay in the code default-off (`memory_v7_write_grammar*`,
  `memory_v7_kind_ce_weights`), the `_gram` config and its launchers were removed. Accepted as generic: rules on the
  model's own behaviour over time, weights driven by its own errors, label-change weights, deployment-matched training.
* 09-16 22:30-23:50 **analysis for v0 (user: talk first, nothing launched):** r5 vs r6 (gradual bank = mirror loop,
  count falls behind, missed 5/10 pick starts at 600; r6 fails only by early notes + press); both use the delta rule
  (r5 rate 0.3 + every-step writes); the surprise (Titans gradient) rule is not recommended (nonlinear bank blurred
  contexts in v6, learned write strength = blending, momentum keeps writing, gates train through the recurrence);
  prefill numbers (window ~half an episode, ~45 % of notes from labels, 6 % of windows start empty); the write has no
  gradient (hard word choice); teacher forcing makes hybrid notes only when the PHASE word is wrong (count-word errors
  are already realistic); per-step CE averages ~10 sentence + ~40 action tokens, so the count word is 1 token in 50.
* 09-17 00:01 **user: "start fresh from the pi05 pretrain base ckpt and everything else as you say ... use the 2h100,
  make full use of the h100"; 00:10 "2h100 is yours, but always keep 1gb alive".** New generic knobs (default off,
  every earlier config unchanged, CPU-checked): `memory_v7_write_retract_steps` (A -> B -> A within K steps erases B by
  subtracting its decayed delta; exact to 6e-8 on the standalone bank; `v7_retracted_count`), `memory_v7_write_vocab_only`
  (candidate must equal a reference sentence token for token; `v7_vocab_rejected_count`), `memory_v7_hard_token_ce_weight`
  (sentence tokens whose argmax misses the label x10, action tokens untouched; `v7_hard_token_count`). Same rules in
  `v5_heldout_video.py` self modes (`--retract-steps`, `--vocab-only`; records `retracted`, summary `retractions`,
  `vocab_rejections`). **v0 launched 00:11** on H100 0,1 of 17422715 (`logs/launch_v0_hgx1.sh`, batch 2, W&B on;
  keep-alive 2220958 untouched); A40 queue `logs/a40_v0_evals.sh` (200/400 self eps 1 3 7; 600/1200 self 7 1 3 0 +
  oracle 7 3). Stop: `logs/stop_v0.sh`. The stale `h100_r5_evals.sh` waiter was stopped.
* 09-17 01:59 **real simulator rollouts for v0 (user: "choose 10 test ep and run real eval").** Model side
  `cluster_robomme/eval/serve.sh` -> `serve_policy.py` -> `scripts/serve_yam_memory.py`; simulator side
  `cluster_robomme/eval/run.sh` -> `rollout.py` (benchmark venv). Changes: `serve_policy.py` now serves any
  `robomme/checkpoints/<config>/<exp>/<step>` whose config starts with `pi05_robomme` (config = grandparent dir);
  `V5SentenceMemory` in `serve_yam_memory.py` got the v7 write rules read from the model (`debounce_steps`,
  `vocab_rows`, `retract_steps` with exact delta retraction through `_canonical_delta_state`), so the served bank
  follows the same rules as training and `v5_heldout_video.py`; the step info carries `retracted`, `retractions`,
  `vocab_rejections`. Driver `robomme/logs/rollout_v0_h100.sh <STEP> <PORT>` (H100 of 17434227, GPU 0 next to the
  1 GB keep-alive; server ~62 GB, so ONE server at a time; `--chunk-size 10` = memory stride; `--max-steps 1300`,
  model seed 7); chain `robomme/logs/chain_v0_rollouts.sh` runs step 400 after step 200. Episode set = the legacy
  balanced val selection, two per pick count: 1,5 | 4,9 | 0,2 | 3,7 | 15,23 (reference B1050 h30: 5/10 = eps 1,4,5,9,15).
  Outputs `robomme/rollouts/<ts>_PickXtimes_ep<NNN>_policy_*/{manifest.json,predictions.jsonl,rollout.mp4}`,
  summaries `robomme/logs/rollout_v0_<STEP>/summary.json`; ~4.5 min per timed-out episode. Viewer page: section
  "Real simulator rollouts" built by the scratchpad `build_rollouts_section.py` (videos `ro_<step>_ep<NN>.mp4`).
* 09-17 02:09 **user: "directly run the same eval on 400 ckpt"; 02:10 "only real rollout and same for all later eval".**
  The step-200 rollout was stopped after 6 complete episodes (1,5,4,9,0,2; ep3 partial) and the offline replay queues
  (A40 `a40_v0_evals2.sh`, H100 battery) were stopped: from now on every v0 checkpoint is judged ONLY by the real
  10-episode rollout. Node-side watcher `robomme/logs/rollout_watch_v0.sh` (iris-hgx-1, job 17434227, log
  `rollout_watch_v0.out`) runs the driver on the newest finished checkpoint not yet evaluated (one server at a time,
  waits for GPU 0 to hold only the keep-alive, port 18700 + step/200); step 400 started 02:11.
* 09-17 02:18-02:23 **execution horizon.** The client executed the first 10 of the model's 50 planned actions per
  query (`--chunk-size 10` = memory tick). Measured on the step-200/400 rollouts: in the episodes where the arm hovers,
  the plan travels only 0.08-0.09 rad (L1, 7 joints) in its first 10 actions but 0.6-0.95 rad by action 30, so with a
  new plan every 10 frames the arm never reaches the part of the plan that moves; the one success (200 ep5) had 0.31
  in the first 10. New `rollout.py --execute-horizon H` (multiple of chunk size, <= 50): the model is still queried
  every 10 frames (memory tick unchanged, every query writes), its actions replace the plan only every H frames; manifest
  field `execute_horizon` (absent = chunk size). **User 02:22: "try horizon 20"** -> the h10 step-400 run was stopped
  at 5/10 episodes (all timeouts: 1,5,4,9,0) and `rollout_watch_v0_h20.sh` (driver `rollout_v0_h100_h30.sh` with
  EXEC_HORIZON=20, logs `rollout_v0_<step>_h20/`, ports 18720+step/200) evaluates 400 and every later checkpoint with
  20 executed actions. Step-200 h10 = 1/6, step-400 h10 = 0/5.
* 09-17 02:37 **v0/400 real rollout, 20 of 50 actions executed (memory queried every 10): 1/10** (5 pressed too early
  = "fail", 4 timeouts; reference B1050 h30 = 5/10). Per episode (picks asked -> notes in the bank): ep5 (1) SUCCESS
  pick1>place>press; ep1 (1) timeout, "pick second" invented, 12 commits / 6 erased by the flip-back rule; ep4 (2) fail
  301 steps pick1>place>PRESS; ep9 (2) fail 840 (670 steps for the first grasp) pick1>place>PRESS; ep0 (3) timeout,
  "pick second" written at step 140 while the first cube was still in hand, then place/pick-second cycle; ep2 (3) fail
  634, early "pick second" during the first carry -> said "third" after the real second pick -> pressed after two;
  ep3 (4) fail 658 pick1>place>PRESS; ep7 (4) timeout, gripper hovered over the cube all episode (motor stall remains
  in some scenes); ep15 (5) timeout pick1>pick2>"all tasks completed"; ep23 (5) fail 419 pick1>pick2>PRESS.
  Horizon 20 fixed the arm in 9/10 scenes (h10: hovering everywhere, 0/5 at 400, 1/6 at 200). What remains is the
  memory decision itself: (a) the model presses after the FIRST place in 4 episodes regardless of the goal count
  (2, 2, 4, 4 picks asked) - the goal count is not driving the press decision; (b) early "pick k+1" notes during the
  carry (before the place) shift the count by one (r6's double count, now closed loop); (c) the pick<->place flip
  between consecutive queries starves the two-in-a-row rule so "second time" is often never written.
* 09-17 03:08 **v0/600 real rollout, horizon 20: 2/10** (both 1-pick episodes; 4 pressed too early, 4 timeouts).
  ep1 324 steps and ep5 393 steps clean (pick1>place>press). Fails: ep4 (2) pressed after the first place while the
  sentence alternated pick-second/place on 7 consecutive queries (never two in a row -> never written); ep0 (3)
  pressed after two picks; ep3 (4) pressed after two picks, the real second "place" note was ERASED by the flip-back
  rule because the model flipped back to "pick second" two queries later; ep15 (5) pressed after the first place.
  Timeouts: ep9 (2) noisy bank of 11 notes ("place" before any pick, "pick first" three times); ep2 and ep7 wrote
  "pick second" before the first pick happened and the arm hovered; ep23 (5) 6 notes, stalled. Step 400 h20 was 1/10.
* 09-17 03:52 **user: "stop eval our 800".** Step-800 rollout stopped after 7 episodes (2/7: eps 1, 5; runaway counts
  in eps 0, 2, 3: "second"/"third"/"fourth" written back to back with no place between, each new note feeding the next
  count; ep4/ep9 pressed after one pick). Watcher `rollout_watch_v0_h20.sh` stopped too; H100 17434227 holds only the
  keep-alive. Training continues on 17422715. Restart later evals with
  `ssh iris-hgx-1 'cd memory_project_robomme && setsid nohup bash robomme/logs/rollout_watch_v0_h20.sh <last_done_step> > robomme/logs/rollout_watch_v0_h20.out 2>&1 &'`.
* 09-17 11:44 **user: "eval newest" / "use the single h100"** -> v0/2800 real rollout (horizon 20) started on 17434227
  (`rollout_v0_2800_h20.out`, port 18734); training at 2820/5000 (~13 s/update, ETA ~19:10). Note: job 17434227 ends
  ~18:32 today (6h48m left at 11:44). Kerberos ticket renewed at 11:43 (valid to 09-18 11:43). Checkpoints on disk:
  600 1200 1800 2400 (permanent) + 2600 2800.
* 09-17 11:59 **v0/2800 real rollout, horizon 20: 2/10** (both 1-pick episodes, 288 and 381 steps; 2 fails, 6 timeouts).
  Multi-pick episodes: ep4 (2) wrote "pick second" after the place, hovered over the cube on the target without
  grasping, then pressed 80 steps later (a note about a pick in progress is treated as a pick done); ep9 (2) ten
  consecutive pick-second/place alternations, fail at 347; ep0 (3) the arm re-grasped the cube 3-4 times but every place
  dropped it beside the target, count reached "fourth"; ep2 (3) cube held above the target from step 210 to the end,
  sentences cycled to "first" again; ep3 (4) "third" and "fourth" back to back at 620/650; ep7 (4) first grasp at 540,
  then second/third/fourth within 200 steps; ep15/ep23 (5) stalls. Run track (horizon 20): 400 = 1/10, 600 = 2/10,
  800 = 2/7 (stopped), 2800 = 2/10. One-pick episodes are solved from 600 on; nothing with 2+ picks has succeeded.
* 09-17 13:15-13:50 **v1 = the bean-scoop recipe on PickXtimes (user: "replicate what we have for the blink bean scoop
  task training, i want to make this task work first").** Diffed `pi05_yam_mem_v5_beansB9` (memory_project_v5) against
  v0 field by field (dumps in the session scratchpad): 32 model / 26 data / 4 training fields differ. Copied: the v5
  Titans MLP sentence bank (hidden 3x1024) with slot keys + whitened values and standardized-attention pooling, read
  query conditioned on the previous sentence (NOTE: this was already ON in v0 too -- the 02:58 statement that it was
  off was wrong), delay 0, retry-until-committed own writes (B), state masking 0.5, TBPTT block 25, prefill 16, stride
  5, slice prob 0.5, min slice 14, buckets 14/27/40, critical pad 75, batch 2 per card, lr 5e-5 (A) / 2.5e-5 (B),
  no debounce / retraction / vocabulary gate / hard-word weight. Not copied: RTC delay 6, max_token_len 80, the YAM
  waiting-state mask, the B6a warm start. LABELS: new version "tgt" = target-carry two-phase sentences
  ("pick up the red cube, 2 of 3" / "place the red cube onto the target, 2 of 3" / press / completed; 92 sentences,
  max 14 tokens, built by `build_pickxtimes_v2_labels.py --label-version tgt`, x cross-checked against every episode
  prompt; dataset `robomme/data/lerobot/PickXtimes_tgt`, spec `prepared_tgt.json`). Configs `pi05_robomme_0917_v1A`
  (label writes, from plain 7000 with fresh memory leaves, 2001 steps, save 250 / keep 500) and `_v1B` (own writes,
  from `OPENPI_ROBOMME_V1A_PARAMS`, default A/1000); `OPENPI_ROBOMME_V1_GPUS` 1 (exact beans, single H100) or 2.
  Launchers prepared, NOT run: `robomme/logs/launch_v1A_h100single.sh` (17434227) / `launch_v1A_h100pair.sh`
  (17422715, would need v0 stopped). Step-0 smoke of v1A on the single H100: `robomme/logs/smoke_v1A.out`.
* 09-17 13:52-14:05 **user questions before starting v1** (MLP vs linear bank; what A/B do; what the slot keys are;
  "memory write every 15 frames so it is 2 hz"). Answered in chat; changes: stride 15 in every v1 config; new
  `_lin` twins `pi05_robomme_0917_v1A_lin` / `_v1B_lin` = the bean recipe (tgt labels, A label writes -> B own
  writes, delay 0, retry, prefill 16, state mask 0.5, block 25, bean sampling, no v7 knobs) on the LINEAR v6.1 bank
  of v0 (`_mem_variant` + `data_overrides`; the twin differs from v0's model only in those fields, checked); the
  plain `_v1A/_v1B` keep the bean MLP bank + slot keys. Single-card smokes of v1A (bean MLP) OOM at batch 2 AND
  batch 1 (a ~40 GB allocation with 67 GB in use; batch-independent), so v1 needs the H100 pair (fsdp 2, one window
  per card like v0 = 78 GB/card). Rollout driver for v1 checkpoints: `robomme/logs/rollout_v1_h100.sh`
  (CONFIG/EXP env, `--chunk-size 15 --execute-horizon ${EXEC_HORIZON:-30}`). Nothing launched.
* 09-17 14:08 **user: (1) keep the 3-layer MLP bank, (2) address it token after token, not the slot rule; analyse the
  token rule before starting.** `_beans_variant` now sets slot keys/whitened values OFF and the v6 token path ON
  (token writes, whitened context keys, context-query pointer read, beta 10) on the bean MLP bank = the v6.0
  combination of 09-08. Evidence on record (cluster_v6/README.md §7, A ckpt 250, 284 lookups): newest note per
  context 71/71 on the MLP bank, older notes 30/71, 17/71, 14/71 (hidden layers pull the context keys together,
  cosine 0.25 -> 0.63); the linear bank with whitened keys 71/71 at every age. PickXtimes count decisions read only
  the newest note per slot. v1A/v1B vs v0 model: MLP bank, block 25, state mask 0.5, delay 0, prefill 16, no v7
  knobs, per-stage oracle/retry flags (CPU-checked). The `_lin` twins stay as the linear alternative.
* 09-17 14:20 `pi0_v6_test.py` 14/14 on CPU after two behaviour-preserving guards in `pi0.py`: the scan called
  `v7_grammar_allows` / `v7_kind_ce_weight` and read `memory_v7_digit_blind*` unconditionally (09-15/16 additions),
  which the tiny test fixtures lack; now skipped (all-ones) when the corresponding knob is empty, `getattr` defaults
  otherwise. Real configs unchanged (their attributes exist; empty grammar/kind lists returned all-ones already).
* 09-17 14:38-14:55 **user: "lets use linear, stop v0, adversarial test, then start training, connect to wandb".**
  v0 stopped 14:42 (step ~3100; ckpts 600/1200/1800/2400 permanent + 2800/3000 on disk). Adversarial pass before
  launch: (a) tgt labels/dataset: every episode contiguous, cycles k=1..x alternating pick/place, x = prompt count,
  one color per episode, press+completed ending, parquet task_index == sidecar (5 episodes), reference tokens
  round-trip through the tokenizer and decode back to the exact sentence, digits are single tokens -> no problems;
  (b) node CPU loader smoke of `pi05_robomme_0917_v1A_lin`: stride 15, 40-step windows, prefill notes = the tgt
  sentences, decision steps counted, 62/96 context tokens; (c) `pi0_v6_test.py` 14/14; (d) client mechanics: one
  v0/2800 episode at tick 15 / 30 executed actions through `rollout_v1_h100.sh` (result in
  `rollout_robomme_0916_v0_2800_h30/`). Vocabulary gate left OFF (bean fidelity; user did not take it up).
  **v1A_lin LAUNCHED 14:5x** on H100 0,1 of 17422715 (`launch_v1A_h100pair.sh`: batch 2 total, fsdp 2, W&B on,
  exp `robomme_0917_v1A_lin`, 2001 updates, ckpt every 250, keep 500). Rollout watcher `rollout_watch_v1.sh` on the
  single H100 (17434227, ends ~18:32) evaluates each checkpoint (tick 15, horizon 30, ports 18750+step/250).
  Viewer builder now labels runs (`<t0> <step>:<label>` windows) so v1 checkpoints do not collide with v0's keys.
* 09-17 15:05 v1A_lin training healthy (both cards 95-99 % util, 77.9 GB each; W&B run d7ybynwv; step-0 ce 12.0
  with the fresh memory path). The one-episode client test against v0 checkpoints cannot run by design:
  `HttpPolicyClient` requires `--chunk-size` == the checkpoint's `memory_stride_frames` (v0 = 10, the v1 driver
  sends 15), so the 15-frame tick is first exercised by the watcher on v1A_lin/250; the horizon-vs-chunk path was
  already proven by the h20 runs. v0 checkpoints left on disk: 600 1200 1800 2400 (permanent), 3000 3200 3400.
* 09-17 15:04-15:12 **user: "do 1000" (cap stage A) + "clean up unused files ckpts".** Cap: node-side
  `robomme/logs/stop_v1A_at_1000.sh` kills the v1A_lin training as soon as checkpoint 1000 is finalized (flat cosine,
  nothing lost; the config still says 2001). Cleanup (`delete_unused_0917.sh`): v0 checkpoints 1200/1800/2400/3000/
  3200/3400 (6 x 27 GB, never rolled out or superseded), the three v1A smoke dirs and the unused `PickXtimes_v2`
  dataset copy -> /iris free 502 -> 642 GB. Kept: v0/600 (27 GB, the evaluated 2/10 reference), r6/600 (27 GB, best
  offline r-line checkpoint), plain 7000 (12 GB, the base of everything), datasets official/tgt/shift1.
* 09-17 16:16 **user: "first 250 is here", "you can use h100 single", "you can stop the current training there"** ->
  v1A_lin training STOPPED at checkpoint 250 (step ~275; ce 12.0 -> 4.65 @100 -> 2.68 @200, memory grad norm
  3112 -> 0.75; the 1000-stopper cancelled; resumable with `run_train.sh` since the ckpt dir holds 250). H100 pair
  17422715 idle (keep-alive only). The 250 rollout runs on the single H100 via the watcher (tick 15, horizon 30).
  Gotcha hit again: a `pgrep -f "...selfwrite_pickxtimes.sh"` inside the ssh command matched the ssh shell itself and
  killed it; use bracket patterns.
* 09-17 16:16-16:30 **misread + correction.** I stopped v1A_lin at 250 reading "stop the current training there" as
  the pair; the user meant the SINGLE H100 ("if anything run on single h100, you can take use of the gpu so you can
  stop there, there means single h100!"). Fixed 16:26-16:30: the trossen step on 17434227 (17434227.432,
  `pi05_base_0917_1h100`) cancelled at the user's word; the 250 rollout restarted there (watcher, 16:27); v1A_lin
  RESUMED on the pair from checkpoint 250 (16:30, W&B run resumed, 1000 stopper re-armed); the partial pair rollout
  (eps 1 ok, 5 ok, 4 fail) discarded. Plan: if the 250 rollout counts right, switch the pair to v1B_lin from 250.

### 2026-09-17 16:48 — v1A_lin/250 rollout on the single H100 restarted (my kill at 16:29)
The 16:27 rollout of checkpoint 250 on the single H100 (job 17434227) died at 16:29:43 with exit 143: a cleanup command
meant for the leftover pair-side rollout matched `rollout.py`/`serve_policy.py` by name without checking the job, so it
killed the single-H100 client and server (2/2 one-pick successes done, episode 4 at step 120). The status message at
16:35 that called it "in progress" was stale. Restarted directly with `rollout_v1_h100.sh 250 18751` at 16:47:34
(T0=1789688854, `rollout_robomme_0917_v1A_lin_250_h30/`); the node watcher (pid 3700510, last=250) is left alive and
will pick up 500 when it lands. Rule: any process kill on iris-hgx-1 filters by SLURM_JOB_ID from /proc/<pid>/environ.
Training on the pair: resumed from 250 at 16:30, ~20 s/update, step 263 at 16:43 (500 ≈ 18:05, 1000 ≈ 20:50).

### 2026-09-17 17:28 — v1A_lin/250: self-write 2/10, forced-sentence 2/10 (diagnostic), replay of training demos
* Self-write rollout (benchmark protocol, tick 15 / horizon 30): 2/10, only the one-pick episodes. Phase agreement with
  the simulator 55% over 338 queries; in 6/10 episodes the model flips to "place" at step 30 (third query) with
  confidence 0.94-0.98 while the arm has barely moved (no demo's first pick ends before step 6.7; median 7.5 steps).
* Forced-sentence rollout (`rollout.py --mode oracle-subtask --oracle-label-version tgt`, new `TgtLabelAdapter`
  rewrites the official online sentence into the tgt vocabulary; server `--diagnostic-oracle-subtask`;
  `robomme/logs/rollout_v1_forced.sh`, videos on the viewer as "v1Aforced"): 2/10 with the TRUE sentence in the bank
  and in the prefix at every step. Every multi-pick episode: first pick and place succeed, the true label moves to
  "pick ... for the second time", the note says "pick up the X cube, 2 of x", and the arm never picks again (button
  press = fail, or stall = timeout). The action expert saw the correct sentence (forced tokens are embedded into the
  causal block before action sampling, pi0.py ~4360) and the correct bank, and still acted on the scene ("cube on
  target -> press"). At 250 updates the policy does not use the count.
* Demo replay (`scripts/v5_heldout_video.py --write-mode self`, `robomme/logs/replay_v1A_250.sh`, 10 training demos
  2 per pick count; there are NO val demos, all 100 demos are split=train): ep12 (1 pick) no early flip on the expert
  video, "place" written 2 steps late, then "pick 1 of 1" again instead of "press"; ep13 said "1 of 1" on a
  three-times prompt. 4 episodes OOMed while the forced server shared the card (rerun queued).
* Training on the pair: step 397 at 17:24, ~15-17 s/update.

### 2026-09-17 17:42 — offline eval on recorded expert val episodes (user: "robomme has expert policy, record 1 per pick count")
* `cluster_robomme/eval/record_expert_val.py` (benchmark venv): builds the val scene with `BenchmarkEnvBuilder`, flips
  every task's `demonstration` flag to True after reset and calls `DemonstrationWrapper.get_demonstration_trajectory()`
  (the benchmark's own motion-planner expert; labels switch at the planner's task boundaries as in the released demos).
  Recorded val episodes 1 (1 pick), 4 (2), 0 (3), 3 (4), 15 (5) -> `robomme/expert_val/PickXtimes_val_ep<NNN>/`
  {frames.npz, meta.json, preview.mp4}; all 5 succeed, 35-90 s each. Diagnostic data only (privileged planner).
* `cluster_robomme/eval/offline_eval.py` + `robomme/logs/offline_eval_v1.sh <STEP> <PORT>`: feeds the recorded frames to
  the normal policy server (own writes) at the memory stride through the same HTTP contract as the real rollout, scores
  the decoded sentence against the true tgt sentence (TgtLabelAdapter). ~4 min per checkpoint on the single H100.
  Output `robomme/logs/offline_<exp>_<step>/{PickXtimes_val_ep*.json, summary.json, offline_eval.log}`.
* v1A_lin/250 offline result: exact 69.6%, phase 92.8%, press timing ok 5/5, early flips 17/192 queries. Ep 1 (x=1)
  100%; ep 4 (x=2) 89% (count reaches 2 of 2, press right after); x=3,4,5: the count stops at "2 of x" for the third and
  later picks (copy of the previous count instead of the increment). NO early "place" at step 30 under expert motion
  (first wrong query >= frame 105 everywhere) -> the step-30 flip of the real rollouts is a closed-loop trigger (arm still
  near home after 1 s under the policy's own slow start), not a data prior.
* `scripts/v5_heldout_video.py` replay of 10 training demos gave garbage decodes ("<loc0758>...", "place green sponge",
  35% exact) -> its input path differs from the served path; not used as evidence. Offline eval = the server path above.
* Queued: `after500_offline.sh` runs the offline eval of 500 right after the watcher's 500 rollout (job ends ~18:32).

### 2026-09-17 17:53 — offline eval gets a stage-A mode: label (oracle) writes, decode scored (user 17:46)
* Server: `serve_policy.py --diagnostic-oracle-write` accepts `oracle_write_subtask` per request; `MemoryPolicy.infer(...,
  oracle_write_subtask=)` decodes freely (returned + scored) but pushes the TRUE sentence through the write rule into the
  bank, and the read queries then condition on the label, as in the stage A training scan. Client
  `HttpPolicyClient.infer_oracle_write`; normal `infer` refuses such a server. `offline_eval.py --write-mode own|oracle`;
  `WRITE_MODE=oracle offline_eval_v1.sh <STEP> <PORT>` -> `offline_<exp>_<step>_oracle/`.
* v1A_lin/250, label writes: exact 88.8% (95/100/84/89/75 for x=3/1/4/2/5), phase 92.8%, press ok 5/5, the count reaches
  x of x in all five (mostly one-query lags at transitions, e.g. "pick 2 of 3" once before "3 of 3"). Same checkpoint
  with own writes: 69.6%, count stuck at 2 from the third pick on -> the count errors come from the polluted own history,
  not from the reader. Stage A at 250 is doing its job on the sentence side; the action side (forced rollout) is not.
* Queued on the single H100 after the watcher's 500 rollout: own-write offline eval (after500_offline.sh) then the
  label-write one (after500_offline_oracle.sh). Viewer: the offline section shows both modes per checkpoint.

### 2026-09-17 18:04 — A stopped at 500, B launched from A/500; single H100 = offline label-writes test only (user 17:58)
* User: "once A lands 500, on 2h100 switch to B; on the single H100 directly run the label-writes test on 500, no real
  rollout". The watcher had already started the real rollout of 500 at 17:55:48; killed it (script/client/server by PID),
  killed the watcher and both after-500 waiters, and the workstation `restart_watcher_1836.sh` (it would have restarted
  the real-rollout watcher inside the pair at 18:36 and, after its 30-min wait, put a server on B's card).
* Pair: `stop_v1A.sh` at 18:00:16 (A reached step 518; checkpoints 250 and 500 kept; the 1000 stopper killed with it);
  `launch_v1B_h100pair.sh` with OPENPI_ROBOMME_V1A_LIN_PARAMS=.../v1A_lin/500/params at 18:00:27 -> robomme_0917_v1B_lin,
  W&B run hl8i6bhq, "Restoring checkpoint from .../v1A_lin/500/params" 18:00:49, first progress tick 18:03:02.
* A/500 label-writes offline test: exact 86.3% (83/95/80/89/84 for x=3/1/4/2/5), phase 89.2%, press ok 5/5, count reaches
  x of x in 5/5. Flat vs 250 (88.8%); new error kinds: "1 of 1" at the very first query in two episodes (x read wrong
  before the first note exists), an x confusion "2 of 3"/"3 of 3" in the 4-pick episode, and a garbage decode
  ("Action: <loc0551>...") once after "all tasks completed" in ep 4 (after the task is over; harmless).
* The single-H100 job 17434227 ends ~18:32; nothing else is queued on it. B checkpoints will need an eval card: the offline
  eval does not need the simulator, so the idle A40 17452297 is a candidate (35 GB server on a 48 GB card, untested).

### 2026-09-17 18:18 — eval card = job 17490595 (user 18:16); B test loop armed; forced A/500 rollout running
* New single H100 job 17490595 (iris-hgx-1, 3-day limit, keep-alive 2904786 untouched). `robomme/logs/watch_v1B.sh`
  runs on it: for every finalized robomme_0917_v1B_lin checkpoint, first the OWN-write offline test on the expert
  recordings (`offline_robomme_0917_v1B_lin_<step>/`), then the real 10-episode rollout (`rollout_robomme_0917_v1B_lin_
  <step>_h30/`); ports 18800+/18820+ step/250. B/250 expected ~19:20.
* Meanwhile the forced-sentence rollout of A/500 (`rollout_robomme_0917_v1A_lin_500_h30_forced/`) runs on the same card
  to see whether the action side follows the count better after 500 A updates than after 250 (2/10 there).

### 2026-09-17 18:26 — forced-sentence rollout of A/500: 2/10, unchanged from A/250
With the true "k of x" sentence forced into the bank and the prefix at every step, A/500 succeeds only on the two
one-pick episodes; in all eight multi-pick episodes the first pick and place succeed and the arm never performs the
second pick (7 fail by pressing the button, ep 7 never even finished the first pick, ep 4 stalled to the step limit).
250 more A updates changed nothing on the action side. Videos on the viewer under "v1Aforced" ckpt 500.

### 2026-09-17 19:38 — B/250 own-write offline test: 73.2% exact, count advances again
Eval card 17490595 (user 19:32: stop their trossen training there, run the eval). B/250 with OWN writes on the 5 expert
recordings: exact 73.2% (51/100/73/82/60 for x=3/1/4/2/5), phase 93.1%, press ok 5/5, count reaches x of x in 4/5
(ep 15 reaches 4 of 5). A/250 own writes was 69.6% with the count stuck at 2. Remaining errors: the first note says
"1 of 1" instead of "1 of x" in three episodes (x read wrong before the first note), colour/target slips in ep 15
("blue" for green, "trash" for target), garbage decode after "all tasks completed" (ep 4). Real rollout of B/250 started 19:37.

### 2026-09-17 19:49 — B/250 real rollout: 2/10 (one-pick episodes only), notes now run AHEAD of the arm
Own-write real rollout of B/250 (tick 15 / horizon 30): ep 1, 5 success; all multi-pick episodes fail. New shape: the
count advances without the arm having done the work (ep 4 "pick 2 of 2" at step 165 while the true second pick starts at
210; ep 23 "3 of 5" at 240 and press at 300 during the true second pick), and the first note reads "1 of 1" instead of
"1 of x" in 6/10 episodes. The arm still does not perform the second pick (true phase reaches "second pick" everywhere,
never "place" again); episodes now last longer (833-1301 steps) because the arm dithers instead of pressing at once.

### 2026-09-17 20:10 — prompt-swap probe: the count word is read only partly, and worse with training
`cluster_robomme/eval/prompt_swap_probe.py` (+ `robomme/logs/probe_prompt_swap.sh`): frame 0 (and 15) of each recorded
expert scene, empty bank, the goal rewritten with each count word (one-pick form for 1; "repeating this action <word>
times" for 2..5); does the decoded "of x" follow the prompt? Results (probes where decoded x == prompt x, of 50):
A/250 39, A/500 27, B/250 28. Every miss says "of 1". "two" is read almost always; "three/four/five" fail in most
scenes at A/500 and B/250; the same count word succeeds in one scene (ep 15 layout) and fails in another (ep 4), so
x is being inferred from the image layout at least as much as from the count word. Prompt path = weak, not dead, and
degrading as the bank-copy shortcut strengthens (75% of training clips start with the label history prefilled; only the
~25% full-trajectory clips start with an empty bank). Fixes proposed (generic): more cold-start clips, higher onset
weight, first-note confirmation at deployment.
* 20:09: user needs the single H100 -> B watcher and all eval processes on 17490595 stopped; evals move to a new
  2xL40S job (`robomme/logs/eval_l40s.sbatch`, partition iris; `watch_v1B_l40s.sh`: offline test on GPU 0, real rollout
  on GPU 1, separate JAX cache dirs; `serve.sh` now honours OPENPI_JAX_CACHE_DIR; eval scripts take EVAL_GPU).

### 2026-09-17 20:16 — B_cs prepared and armed (user 20:12 "yes do it")
* New config `pi05_robomme_0917_v1B_lin_cs` = v1B_lin continued from B/500 params (env OPENPI_ROBOMME_V1B_LIN_PARAMS)
  with: data memory_slice_prob 0.5->0.3 and memory_critical_prob 0.5->0.3 (full-trajectory clips that start at frame 0
  with an empty bank: 25% -> 49%; transition-anchored 50% -> 30%; random slices 25% -> 21%); model
  memory_v7_onset_ce_weight 1 -> 3; new serving rule memory_v7_first_write_debounce_steps=2 (while the bank is empty a
  sentence must be decoded twice in a row before it is written; `V5SentenceMemory(first_write_debounce_steps=)`,
  new Pi0Config field, serving only). Verified on CPU: only these diffs vs v1B_lin; loader = B/500.
* Armed: `after_b500_switch_ws.sh` (workstation) waits for both B/500 tests (L40S watcher), then on iris-hgx-1 runs
  `stop_v1B.sh` (patterns with a trailing space so B_cs never matches) and `launch_v1B_cs_h100pair.sh` on the pair.
  `watch_v1Bcs_l40s.sh` runs inside the L40S job (ports 18840+/18860+) and tests every B_cs checkpoint (offline own-write
  on GPU 0, real rollout on GPU 1). Viewer builder knows the run as "v1Bcs".

### 2026-09-17 20:52 — L40S job 17492232: card 1 unusable, card 0 needs the platform allocator
* GPU 1 of the job (iris9, UUID GPU-d7b01cd6-5ece-3ad2-2c4a-a9100a70ead0): JAX "Unable to initialize backend 'cuda':
  no supported devices found", torch sees it but a 1-element allocation hangs; nvidia-smi lists it healthy. Not used.
* GPU 0: the fp32 server OOMed at the first inference with the default XLA pool allocator (46 GB card); with
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` it runs at ~27 GB. Set by default in rollout_v1_h100.sh / rollout_v1_forced.sh
  and exported by `watch_v1B_l40s_seq.sh` (one-card sequential watcher: offline test, then rollout, for B and B_cs;
  PORTBASE 18800 / 18840). The parallel two-card watcher is retired. B/500: offline retry running, rollout chained
  after it (`after_offline500_rollout.sh`); the B -> B_cs switch on the pair waits for both DONE files.
* 20:55 (user 20:53 "use single h100 to run eval"): evals back on job 17490595. `watch_v1_seq.sh` (generic one-card
  watcher; allocator/cache from the env, defaults: XLA default allocator + robomme/cache/eval_jax) runs for B (last=500)
  and B_cs (last=0); `after_offline500_rollout_h100.sh` runs the B/500 rollout on the H100 as soon as the L40S offline
  test of B/500 (already 4/5 scenes) writes DONE. L40S job to be cancelled afterwards.

### 2026-09-17 20:57 — B/500 own-write offline test: 73.9% (B/250 73.2%), count reaches x of x in 5/5, first note still "1 of 1"
exact 76/100/49/68/77 for x=3/1/4/2/5, phase 96.0%, press ok 5/5, early flips 11. The first note says "1 of 1" in 3 of the
4 multi-pick scenes (same as B/250 and the probe); mid-episode x slips ("2 of 3" in the 4-pick scene, "2 of 7" once) and a
colour slip ("red cube" for green in ep 15). B/500 real rollout started 20:57 on the H100; the B -> B_cs switch follows.

### 2026-09-17 21:07 — B/500 real rollout: 2/10, but the arm completes a second cycle for the first time
Own-write rollout of B/500 (H100, tick 15 / horizon 30): ep 1, 5 success; all multi-pick fail. New: in ep 0 (3 picks) and
ep 23 (5 picks) the simulator's true phase reaches "second pick -> place -> third pick" (the arm did a second pick-and-
place; never happened at A/250, A/500 or B/250), then the run fails by an early press (ep 23: note "3 of 5", press at
345). Notes still start with "1 of 1" in 7/10 episodes and jump ("3 of 2", "2 of 1"). The B -> B_cs switch fired 21:06:57.

### 2026-09-17 22:41 — B_cs/250 own-write offline test: 79.0% (B/500 73.9%), first note right in 2 of 4 multi-pick scenes
exact 51/100/67/96/81 for x=3/1/4/2/5, phase 96.4%, press ok 5/5, count reaches x of x 5/5, early flips 10. First note:
ep 4 "1 of 2" and ep 15 "1 of 5" right; ep 0 and ep 3 still "1 of 1" (B/500 had 3 of 4 wrong). Served with the first-note
confirmation (memory_v7_first_write_debounce_steps=2). Real rollout of B_cs/250 started 22:41 on the H100. B_cs training:
step 272, ce 0.99 / flow 0.0072 at step 200.

### 2026-09-17 22:55 — new viewer page for the v1 line; B_cs/250 real rollout 1/10
* User: "previous is too long, new page only show start from v1" -> https://claude.ai/artifact/KFwEp9Pxz8kZhy7jAQus5R
  (checkpoint summary table + offline section + real/forced rollouts, v1A/v1B/v1Bcs only; 65 videos, 28 MB). The old
  page 2tJC79PFv2Q8E5B4NNibAG stays as the archive. Builders: build_rollouts_section.py --only v1, build_offline_section.py,
  build_v1_summary.py (scratchpad viewer_v1/).
* B_cs/250 real rollout: 1/10 (ep 5). ep 1 (one pick) timed out: the arm never grasped while the notes flipped
  place/PRESS; ep 3 (4 picks): the arm completed two cycles before failing. First note "1 of 1" in 4 of 9 multi-pick
  episodes (B/500: 7/10). Many more notes per episode (10-16 vs 3-6): closed-loop flipping is up, not down, at this
  early checkpoint. Offline own-write test of the same checkpoint: 79.0%.

### 2026-09-18 00:45 — B_cs/500: offline 79.5% (own writes), real rollout 2/10; page KFwEp9Pxz8kZhy7jAQus5R v2
- Offline own writes (`offline_robomme_0917_v1B_lin_cs_500/`): mean exact 79.5% (B/500 73.9, B_cs/250 79.0); count reached 5/5;
  first note "of 1" in 3 of 4 multi-pick scenes (ep004 corrected itself at frame 45); early flips 11 (B/500 11, B_cs/250 10).
- Real rollout (`rollout_robomme_0917_v1B_lin_cs_500_h30/`): SUCCESS 2/10, the two one-pick episodes, same as B/500.
  Multi-pick: first note "of 1" in 4 of 8 (B/500: 6 of 8). Arm never got past "pick for the second time" in any
  multi-pick episode; in 6 of 8 the notes counted up to "k of k" and PRESS while the arm had placed once, so the arm
  pressed early (= fail). ep 23 (five picks) timed out with 30 notes cycling "1 of 5" pick/place, arm stuck after the
  first place. Notes per episode 7-30 (B/500: 5-39).
- Reading: B_cs improves the first note a little and the offline score a little, but in closed loop the count still
  runs ahead of the arm (the writer counts a place the arm did not finish). Success unchanged at 2/10.
- Page rebuild is now one command: scratchpad `rebuild_v1_page.sh` (rollouts --only v1, all offline dirs, summary).
- B_cs training at step ~660 (17.5 s/update); 750 ≈ 01:05, 1000 ≈ 02:20; watcher on 17490595 tests each.

### 2026-09-18 01:15 — B_cs/500 label-write offline test 91.2%; forced-run frames show the arm pressing with perfect notes
- `offline_robomme_0917_v1B_lin_cs_500_oracle/` (true sentence written, decode scored): mean exact 91.2% (A/250 88.8, A/500 86.3),
  count 5/5, early flips 16; first query still "1 of 1" in 3/4 multi-pick scenes (empty bank at frame 0, prompt count not read).
  Run by hand on 17490595 (port 18890) while the watcher waited for the card; page rebuild script now includes the dir.
- Forced A/500 last frames (scratchpad forced500_ep0{00,09}_last.jpg): with the note "pick up the blue cube, 2 of x" in the
  bank the gripper is over the button in 8/10 episodes -> the arm ignores the note after the first place and presses.
  Training data would not favour that: after a place the expert re-picks 167x and presses 100x (37%). Demos per x:
  1:23 2:25 3:27 4:12 5:13.
- Hypothesis noted for the user: in stage B a wrong own note + expert action target teaches the arm to ignore the note.
  Proposed cheap check (awaiting the user's go): forced rollout on B/500 and B_cs/500 to see if the arm reads notes better after B.

### 2026-09-18 01:13 — forced-sentence rollouts of B/500 and B_cs/500 queued on the eval card (user "ok do it")
- `robomme/logs/forced_B_queue.sh` (log `forced_B_queue.log`, job 17490595): waits for the GPU to be free of other servers
  (the watcher's 750 tests run first), then `rollout_v1_forced.sh` for B/500 (port 18895) and B_cs/500 (port 18896);
  results in `rollout_robomme_0917_v1B_lin[_cs]_500_h30_forced/`. Question: does the arm read the note better after B?
  Baseline forced A/250 2/10, A/500 2/10.

### 2026-09-18 01:30 — B_cs/750: offline own 79.2%, real 2/10 (page v5)
- Offline own writes flat since 250 (79.0 / 79.5 / 79.2); count 5/5; four-pick scene still the weak one (53%).
- Real rollout `rollout_robomme_0917_v1B_lin_cs_750_h30/`: SUCCESS 2/10 (one-pick only). First note "of 1" in 5/8
  multi-pick; arm reached "pick second time" in 7/8 and then pressed on a "k of k / PRESS" note; ep 15 stuck after the
  first place (20 notes, timeout). Same closed-loop pattern as B/500 and B_cs/500. Forced B/500 started 01:24:47.

### 2026-09-18 01:45 — forced B/500: 2/10 by episode status (the .out prints "SUCCESS 0/10" because diagnostic runs leave
`success` unset; count `status == success` in the manifests, as the page does). Arm behaviour with PERFECT notes, 8 multi-pick eps:
- A/500: pressed the button after the first place in 6/8 (one gripper close over the button = fail), 2/8 timeouts with
  repeated failed grasp attempts. B/500: pressed in 4/8, 4/8 timeouts hovering over the cube on the target with 5-7 failed
  grasp closes (ep 3: no gripper action at all). So B training moved the decision from "press" toward "re-pick", but the
  arm never managed a second grasp from the target in any forced episode (real B/500 had 2 second cycles).
- Several episodes close the gripper at identical frames (993, 1053): the replan at a chunk boundary starts a close and
  aborts -> indecision at each tick, not a steady plan. Frames: scratchpad forcedB500_ep00{0,4}_last.jpg.
- Bottleneck chain under perfect notes: decision (press vs re-pick, improving A->B) then re-grasp skill (0 for N so far).

### 2026-09-18 01:50 — forced B_cs/500: 2/10; arm trend flat after B (page v7)
- `rollout_robomme_0917_v1B_lin_cs_500_h30_forced/`: 2/10 by status. Multi-pick: pressed after the first place 4/8
  (ep 9, 2, 15 quickly; ep 7 after 800 frames of hovering), timeouts 4/8 (ep 4 six failed grasp closes, ep 3 two, ep 0 and
  ep 23 no gripper action). Second grasp from the target: 0/8 (A/500 0, B/500 0). Queue ended 01:42; eval card idle,
  watcher waits for B_cs/1000 (~02:20).
- Forced trend, 8 multi-pick eps: pressed 6 (A/500) -> 4 (B/500) -> 4 (B_cs/500); re-pick attempts 2 -> 4 -> 4; re-grasp 0 -> 0 -> 0.

### 2026-09-18 02:00 — CORRECTION + where the arm goes after the first place (joint-space classification of gripper closes)
- Each gripper close after the first place was classified by joint distance to that scene's button pose (from registered
  presses) vs its target pose (release of pick 1). Wrist contact sheets confirm two cases (scratchpad grasp_*_sheet.jpg):
  B_cs/500 forced ep 4 "6 closes" are all AT THE BUTTON (presses that did not register -> timeout), not grasp attempts;
  B/500 forced ep 15 first close is around the cube on the target, later closes drift between cube and button.
- Closes after place 1, 8 multi-pick episodes, button / cube-on-target:
  forced A/500 12 / 3 · forced B/500 3 / 20 · forced B_cs/500 9 / 3 · real B/500 5 / 9 · real B_cs/750 6 / 6.
  Reading: with perfect notes A presses; B goes for the cube (but never grasps it, closes misaligned); B_cs goes back to
  pressing. Consistent with "more own-write clips in training (B_cs 49% cold starts vs B 25%) teach the arm to trust the
  note less" (hypothesis, not proven). Re-grasp from the target succeeded 0 times in any forced run; 2 second cycles ever (real B/500).
- The 01:45 entry's "hover/failed grasp" reading for the B_cs/500 timeouts was wrong: they were ineffective presses.

### 2026-09-18 02:26 — B_cs/1000 landed; user "1k is there analysis first"
- Watcher: offline own-write test started 02:24:38, real rollout follows. Forced rollout of 1000 queued behind them
  (`forced_Bcs1000_queue.sh`, port 18897, log `forced_Bcs1000_queue.log`). No decision on the arm options until these are in.

### 2026-09-18 02:45 — B_cs/1000: offline own 85.1% (first note right 3/4), real 1/10
- `offline_robomme_0917_v1B_lin_cs_1000/`: 85.1% (750: 79.2), count 5/5, first note right in 3/4 multi-pick scenes, early flips 13.
- `rollout_robomme_0917_v1B_lin_cs_1000_h30/`: SUCCESS 1/10. ep 1 (one pick) timed out before the first place (2 notes,
  arm never placed); ep 9 (two picks) timed out at the first pick with 41 flipping notes. Multi-pick: first note "of 1"
  3/8 (750: 5/8); after the first place the arm pressed on an early "k of k"/PRESS note as before (closes: 6 button / 4 cube);
  no second cycle. Forced 1000 started 02:41:32.
- Code reading for the user's question (02:35): the causal block in training holds the LABEL sentence (teacher forcing)
  and the action suffix attends to it (`_v32_suffix_mask`: early/late views both include causal_mask), so the arm always
  sees the true sentence while training; only the 8 bank-read tokens can carry a wrong own note in stage B. No press
  upsample exists (transition-anchored branch weights every change equally; onset weight 3 is on the sentence CE only).
  Proposed (not launched): (1) 10-min diagnostic = force "press the button to stop" from tick 0 (does the arm read the
  sentence at all?); (2) action-loss onset weight at label-change steps (+ optional per-step flow mask when the bank note != label).

### 2026-09-18 02:47 — arm diagnostic queued: "press the button to stop" forced from tick 0 (user "yes run")
- `eval/rollout.py`: new `--mode oracle-fixed --oracle-fixed-sentence "<sentence>"` (default "press the button to stop");
  same server flag as the label-forced runs (`--diagnostic-oracle-subtask`), the fixed sentence is written to the bank and
  embedded before action sampling at EVERY tick; manifest `mode="oracle-fixed"`, field `oracle_fixed_sentence`,
  `diagnostic_success`; the page builders ignore this mode (never mixed with the label-forced runs).
- `robomme/logs/rollout_v1_forced_fixed.sh` (log dir suffix `_forced_press0`), queue `forced_press0_queue.sh`
  (log `forced_press0_queue.log`): B_cs/1000 (port 18898) then B/500 (18899), behind the forced 1000 run.
- Reading: if the arm goes to the button before touching the cube, it reads the sentence and the failure is specific to
  the re-pick decision; if it picks the cube as usual, the arm ignores the sentence entirely.

### 2026-09-18 02:55 — forced B_cs/1000: 2/10 but the arm STOPPED pressing early; first forced multi-pick success (ep 4)
- `rollout_robomme_0917_v1B_lin_cs_1000_h30_forced/`: ep 4 (two picks) completed pick 2 / place 2 / press = success (first
  ever in a forced run); ep 5 success; ep 1 (one pick) timed out at the FIRST pick (arm quality, same as the real run);
  ep 9 pressed early (fail); the other 6 multi-pick episodes TIMED OUT hovering after the first place with the true note
  "pick 2 of x" (ep 0/2/3/15 no gripper close at all, ep 7 two closes at the cube, ep 23 nine at the cube).
- Trend with perfect notes, 8 multi-pick eps, episodes ending in an early press: A/500 6 -> B/500 4 -> B_cs/500 4 -> B_cs/1000 1.
  Closes after place 1 (button/cube): 12/3 -> 3/20 -> 9/3 -> 2/11. Re-grasp landed: 0 -> 0 -> 0 -> 1.
  So B-stage training IS moving the arm's decision (my 02:00 "flat" reading was premature); the remaining wall is the
  re-grasp from the target (hover, no close) plus a pick-1 failure at 1000 (ep 1). Press-from-tick-0 diagnostic running.

### 2026-09-18 03:05 — press-from-tick-0 diagnostic on B_cs/1000: the arm picked the cube first in 10/10
- `rollout_robomme_0917_v1B_lin_cs_1000_h30_forced_press0/`: with "press the button to stop" in the bank and in front of
  the action tokens from tick 0, every episode still started with pick 1 and place 1 (env phases), and only then pressed
  (ep 0/4/7/23 fail by press after place 1; ep 5 x=1 success; ep 1 and 9 timed out, ep 1 again failing around the first
  pick as in the real/forced 1000 runs). Reading: when the picture strongly disagrees with the note (cube on the table,
  arm at home, note says press) the picture wins; the note acts as a tie-breaker where the picture is ambiguous (after a
  place), which is where the forced trend (early presses 6->4->4->1) shows it working better with more B training.
  B/500 press0 running (started 03:02).

### 2026-09-18 03:10 — press-from-tick-0 on B/500: 9/10 picked the cube first (ep 9 never grasped); same reading as B_cs/1000.
- `rollout_robomme_0917_v1B_lin_500_h30_forced_press0/`. Queue ended 03:09; eval card free. Next: watcher tests 1250
  (~03:35), forced 1250 queued behind them (`forced_Bcs1250_queue.sh`, waits for the watcher's real-rollout DONE).

### 2026-09-18 03:55 — B_cs stopped; NEW BASE `pi05_robomme_0918_base_kiP_off` launched on the pair (user 03:43 + 03:46)
- User: stop the 2xH100 training now; train a new pi05 with knowledge insulation and the subtask, with the subtask in the
  prompt so the arm follows the prompt; 10000 updates; then the memory stages on top; use the OFFICIAL subtask labels.
- B_cs stopped 03:44 (`stop_v1B_cs.sh`, pid 3054365 + launchers; keep-alives untouched). Last B_cs checkpoint 1250;
  the eval card still runs its 1250 tests and the forced 1250 queue.
- Config = `_base_ki_config(official)` + v7 phase-context prompt fields (prompt_prev_subtask stride 15, dropout 0.1 ->
  "none"), max_token_len 256, cosine 2.5e-5 -> 2.5e-6 over 10k (warmup 200), EMA 0.999, batch 16 fsdp 2, save every 1000,
  keep 2000-multiples + 2 newest, W&B on, warm start = plain 7000. Prompt seen by the model:
  "Task: <goal>, State: <8 digits>, Last: <official sentence 15 frames earlier | none>;" then the KI head decodes the CURRENT
  sentence + FAST tokens (CE trains the VLM), flow trains the action expert against a stop-gradient prefix.
- Launcher `robomme/logs/launch_base_kiP_h100pair.sh` (EXP robomme_0918_base_kiP_off_r1); logs train_<exp>.log (+ W&B
  output.log). Not yet done for the memory stage: the memory sequence path does not support prompt_prev_subtask
  (NotImplementedError in config.py) -> the slot must be fed from the committed note inside the training scan; and the
  robomme serving path must send "Last" = the newest committed note (or "none").
- 03:53 first steps: 1.6-1.8 it/s at batch 16 (W&B jwh8yiji) -> 10k in ~1.6 h, done ~05:30. Step 0: ce 10.96 (fresh
  sentence head), flow 0.008 (already an action policy), grad norm 71 (clipped to 1).

### 2026-09-18 04:10 — v7 PROMPT SLOT built; base restarted as r2 with the slot; v2 A/B configs added (user 03:54/03:57)
- User: once the base is ready, train v2 stage A then B to make the model strongly follow the prompt and subtask; fully
  remove the visual bank (no injected columns).
- Token convention (one for base, v2 loader, v2 model, serving): context = bos "Task: <goal>, State: <8 digits>, Last:" +
  a 16-position slot holding the previous sentence's STANDALONE ids (same ids as the causal decode / sentence bank rows,
  no newline; pads masked) + ";\n". Checked on CPU: whole-string encode == head + " sentence" + tail, and standalone ids
  differ from the leading-space ids only at the first token -> the slot uses standalone ids everywhere, hence base r2.
  Code: `FASTSubtaskTokenizer(prev_slot_len)`, `_encode_context`, `slot_null_row`; `tokenize()` / `tokenize_split()`;
  `TokenizeMemorySubtaskInputs(prev_subtask, prev_dropout, prev_null)` emits `token_slot_mask` + the null row;
  Observation fields `token_slot_mask/prompt_slot_null_tokens/prompt_slot_null_mask`; Pi0Config `prompt_slot_len`,
  `prompt_slot_dropout`, `memory_v7_prompt_slot_from_bank`, `memory_v7_prompt_slot_bank_dropout`,
  `memory_v7_no_visual_block`; pi0.py `v7_apply_prompt_slot` (stage-B overwrite before embed_prefix in the scan; unit
  tested), `_memory_token_total`/block assembly/sem_start honour no_visual_block; serve_yam_memory sends
  `prev_subtask` = newest committed note or "none" when the model has a slot. Tests: robomme/logs/test_prompt_slot_loader.py (node CPU).
- Base r1 (text form) stopped at ~1100 (04:07); r2 = same config + prompt_slot_len 16, launched 04:08, 10k ~05:45.
- v2 configs: `pi05_robomme_0918_v2A_off` (label writes, 500 updates, from base r2/10000, official sentences, slot from the
  previous label, dropout 0.1, no visual block, max_token_len 112) and `pi05_robomme_0918_v2B_off` (own writes from A/500,
  B_cs mix, slot from the bank + bank dropout 0.1, 2000 updates). Chain `robomme/logs/chain_v2_h100pair.sh` (waits for the
  base 10000, A, then B). Not yet: a step-0 GPU smoke of v2A (planned on the eval card), the eval watcher for v2.
- 04:09-04:15 BUG + FIX: the serve_yam_memory prompt-slot patch had put `_newest_note_text` INSIDE `MemoryPolicy.__init__`
  (the rest of __init__ became dead code -> "'MemoryPolicy' object has no attribute '_lock'"), so the watcher's B_cs/1250
  offline test, real rollout and forced run all errored (10 episodes state=error, builders skip them). Fixed 04:13
  (method at class level), failed outputs renamed *.failed.out, `rerun_1250.sh` re-runs offline -> real -> forced on
  the eval card; `smoke_v2.sh` then smokes v2A (from base r2/1000) and v2B (from the A smoke checkpoint) at batch 1 there.
- base r2 at 04:13: step 332, 1.8 it/s, ce 10.96 -> 2.50 (step 300), flow 0.004, no token truncation warnings.

### 2026-09-18 04:30 — B_cs/1250 (re-run after the server fix): offline own 82.4%, real 1/10 (page v11)
- Offline own writes: 82.4% (1000: 85.1), count 5/5, first note right 2/4, early flips 10.
- Real: 1/10; the one-pick blue episode timed out at the first pick for the third checkpoint in a row (1000 real,
  1000 forced, 1250); the other multi-pick episodes fail as before (early press or note storms: ep 9 41 notes).
  First note "of 1" 4/8. No second cycle. Forced 1250 running next, then the v2 smoke. This closes the v1 line unless
  the user asks for more; the pair now trains the v2 base.
- 04:37 forced B_cs/1250: 1/10 (ep 1 failed during the first pick), early presses back to 6/8 (1000: 1/8), closes after
  place 1 button/cube 6/8. The 1000 forced result was not a monotone trend. v1 line closed. v2 smoke started 04:37.
- 04:40 smoke v2A (batch 1, eval card): loader OK (matched 51, fresh 108 from base r2/1000), then the sequence collate
  rejected the new per-step field ("unregistered temporal sequence fields: ['token_slot_mask']") -> registered it in
  data_loader._SEQUENCE_TIME_KEYS; smoke relaunched (smoke_v2_run2.out).
- 04:43 smoke run 2: jaxtyping rejected `prompt_slot_null_tokens` ("*sb w" reused the image-width name "w") -> renamed
  the slot-width dim to `slot_w` in Observation; smoke run 3 launched (smoke_v2_run3.out).
- 04:50 smoke run 3 (v2A, one card): loader, data pipeline, type checks and COMPILE all passed; the first update then
  OOM'd (44.5 GiB op; the known 40-step-window limit at fsdp 1). Treated as a pass for the A path. Stage-B path traced
  separately with `pi05_robomme_0918_v2B_off_smoke` (v2B settings, memory leaves fresh from the base) via
  `smoke_v2B_trace.sh`: pass = compiles.
- 04:53 the first stage-B trace smoke died at weight loading: base r2's checkpoint 1000 had been rotated away by the keep
  policy (keep_period 2000 + max_to_keep 2 -> 2000 and 3000 remain). `smoke_v2B_trace.sh` now picks the newest finalized
  base checkpoint; relaunched (smoke_v2B_trace2.out).
- 04:59 stage-B trace smoke (from base r2/3000): loader matched 51 / fresh 108, data pipeline and COMPILE passed, first
  update OOM'd on one card (45 GiB op) exactly like the A smoke -> both v2 paths are shape-valid; the fsdp-2 run on the
  pair (the chain) is the memory test. Smoke checkpoint dirs removed.

### 2026-09-18 07:38 — base r2 DONE (exit 0 at 06:04, final ckpt 9999, ce 0.29 / flow 0.0019 at 9900); chain relaunched
- MISTAKE: the chain waited for checkpoint 10000, but an N-step run's final checkpoint is N-1 (9999; see the memory note
  "checkpoint keep_period deletes 999"). The pair idled 06:04-07:38. Fixed: v2 configs and the chain now use r2/9999
  (`OPENPI_ROBOMME_V2_BASE_PARAMS` exported in the chain); stuck chain killed, relaunched 07:38 (chain_v2_run2.out).
  Base checkpoints kept: 2000/4000/6000/8000 (keep_period) + 9000 + 9999.
- 07:40 relaunch tangle: an inline ssh `pgrep/kill -f chain_v2_h100pair` killed my own shell (self-match) and left the
  state unclear; `robomme/logs/chain_ctl.sh status|start|stop` (anchored pattern, run as a script) now manages the single
  chain instance. One chain (pid 927501, 07:39:44) waits for r2/9999 -> stage A.
- 07:53 v2A running on the pair (W&B iotqmrf6): matched 51 / fresh 108 from r2/9999, compile ~8 min, 77.9 GB per card
  at 95% util (fits, near the limit), step 0 ce 7.74. First steps 150-260 s/it while the three sequence buckets compile;
  steady rate to be confirmed (v1 was ~17 s/update).
- 08:02 v2A steady rate ~12 s/update (steps 19-21; v1 was ~17 s, fewer tokens without the visual block): 500 updates
  ≈ 100 min -> v2A/250 ≈ 08:35, v2A/500 ≈ 09:25, then stage B (2000 updates ≈ 6.7 h -> ≈ 16:10).
- 08:58 v2A/250 landed (effective rate 16.6 s/update, so v2A/500 ≈ 10:05); losses: step 100 ce 0.92 / flow 0.023,
  step 200 ce 0.70 / flow 0.012 (base ended at flow 0.002). Watcher started the label-writes offline test 08:58:26;
  own-writes, forced and real follow (all four ≈ 09:25).
- 09:00 v2A/250 label-writes test: HTTP 500 -> `_newest_note_text` iterated a [1, 48] row (TypeError); fixed with
  reshape(-1). Watcher stopped (`watch_ctl.sh`), failed 250 outputs renamed *.failed.out, watcher restarted 09:04 -> it
  reruns all four v2A/250 tests.
- 09:05 v2A/250 offline LABEL writes (official sentences): mean exact 96.8% (v1A/250: 88.8), first note right in 5/5
  scenes (the official first sentence needs no count, so the "of 1" problem is gone by construction), early flips 8.
  Count metrics are n/a for official labels (summary builder updated). Own-writes test running, then forced and real.

### 2026-09-18 09:27 — v2A/250: offline label 96.8% / own 92.2%; forced 1/10 but multi-cycle arm; real 1/10
- Offline own writes 92.2% (v1A/250: 69.6; best v1 ever 85.1), first note right 5/5, early flips 6.
- Forced (true official note): status 1/10, BUT ep 15 (five picks) completed three full cycles and ep 3 (four picks)
  two before failing -- no v1 checkpoint ever did more than one second cycle under perfect notes. Early presses 3/8
  (ep 4, 3, 23); hover at the cube ep 7, 15. Arm-skill regressions at 250: ep 1 failed the place, ep 9 the first pick,
  ep 0 timed out at the first pick (flow loss 0.012 vs base 0.002 -- the memory injection perturbs the arm; watch 500).
- Real (own notes): 1/10; notes mostly well ordered (ep 4 perfect sequence) but still run ahead of the arm (press written
  before the second place) -> early press; ep 9 "pick > press"; ep 0 stuck at the first pick. Page v13.

### 2026-09-18 09:58 — v2A done (500, exit 0; step 500 ce 0.55 / flow 0.008), v2B started 09:58:09 from A/500
- v2A losses: step 300 ce 0.65 / flow 0.0098, 400 0.62 / 0.0087, 500 0.55 / 0.0080 (base ended at flow 0.002).
- Chain moved to stage B (`robomme_0918_v2B_off`, own writes, slot from the bank, 2000 updates ≈ 9 h at 16.6 s/update
  -> ≈ 19:00). Watcher picks up A/500 (label/own offline, forced, real) next.
- 10:11 v2B running (W&B rthoqcra): all 159 leaves from A/500, step 0 ce 0.82 / flow 0.0062, ~12 s/update after compile.

### 2026-09-18 10:30 — v2A/500: offline label 96.0 / own 88.7; forced 2/10 with FULL sequences; real 1/10 (page v14)
- Forced (true notes): ep 3 (four picks) completed all four places and reached the press phase, ep 4 (two picks) both
  places and the press phase -- both then TIMED OUT pressing the button without the press registering (ep 4: six closes
  at the button); ep 7 and 15 three places each. Places completed per multi-pick episode [2,0,0,1,4,3,3,1] = 14
  (v1 forced never exceeded ~10). Skill regressions remain in 3-4 scenes (first pick fails/stalls: ep 9, 0; hover ep 2, 23).
- Real (own notes): 1/10; places per multi-pick episode [1,0,0,1,2,2,2,2]; notes mostly in order but with repeats and
  "press" written before the last place (ep 3, 4, 2) -> early press. Stage B (own writes) is the intended fix.

### 2026-09-18 11:44 — v2B/250: FORCED 6/10, REAL 3/10 (first multi-pick real successes), offline own 94.9% (page v15)
- Offline own writes 94.9% (best ever), first note right 5/5, early flips 4, 4-12 notes per scene.
- Forced (true notes): 6/10 by manifest status -- both one-pick, the two-pick green, the three-pick green, BOTH four-pick
  scenes (all four places + press registered); the five-pick blue did 4 places and reached pick 5 before the time limit.
  Places per multi-pick episode [2,1,0,3,4,4,1,4] = 19 (v2A/500: 14, v1 best ~10). Failures: ep 9 (place) and ep 15 fail,
  ep 0 stuck at the first pick (this blue three-pick scene fails at pick 1 in every v2 checkpoint so far).
- Real (own notes): 3/10 = one-pick x2 + the two-pick green (first multi-pick real success in either line). Failures:
  notes still run ahead in 4 scenes ("press" after the second pick: ep 2, 15, 23; ep 7 flipping), the four-pick blue did
  all four places but the notes under-counted (stuck at "second time") and it failed at the end; ep 0 stuck at pick 1.
- The forced .out "SUCCESS n/10" line undercounts diagnostic runs; the page/README use manifest status.

### 2026-09-18 13:56 — v2B/500 tests interrupted by an outside SIGTERM; watcher restarted
- The offline own-writes test of v2B/500 finished normally at 12:23 (MEAN exact 92.3%, phase 97.6%, press ok 5/5,
  vs 94.9% at 250). The real rollout was killed by SIGTERM at 12:27:26 after 5 episodes (rc=143; eps 1 and 5 ok, 4 and 9
  failed, 0 running) and the forced rollout plus the watcher itself at 12:33:50 ("Terminated"). No process of ours sends
  signals, no slurm step, login or job submission at those times, the eval job 17490595 kept running; the sender is
  unknown. Partial outputs moved to robomme/logs/killed_0918_1227/. Watcher restarted 13:55 (pid 1117715, watch_ctl.sh
  start); it re-runs the real and forced tests of 500, then 750 (landed ~13:20) and later checkpoints.
- Training was untouched: v2B at update 849/2000 at 13:46, 13.5 s/update, end ~18:00.
- Two obsolete v1 watcher loops (watch_v1_seq.sh 0 / 500, pids 3046580/3046581, started 09-17 20:55) are still sleeping
  on the node; harmless (no v1 checkpoint will appear), not stopped because the stop was refused by the permission
  classifier together with the restart.

### 2026-09-18 14:22 — v2B/500 (rerun after the kill): FORCED 2/10, REAL 2/10, offline own 92.3% — worse than 250 on all three
- Forced (correct notes): only the one-pick scenes. End frames of the rule failures (scratchpad frames500/): ep 3 (x4),
  ep 7 (x4), ep 15 (x5) = the arm pressed the button while the note said "pick up ... for the 3rd/4th time" (early press);
  ep 4 (x2) = the arm grabbed the cube while the note said "press". ep 2 (x3) held the cube for ~1200 frames without
  placing; ep 9 and ep 0 never finished the first grasp; ep 23 stalled at pick 2. So at 500 the arm again acts against a
  correct note in 4 scenes, the v1 failure that 250 did not show (250 forced 6/10).
- Real: 2/10 (one-pick scenes). Notes ran ahead in ep 2/4/9/15 ("press" after 1-2 places), stuck at "second" in ep 3/23.
- Training curve is smooth (ce 0.82 -> 0.34 at 500 -> 0.32 at 900; flow 0.006 flat), so no instability; either the
  forced test has large checkpoint-to-checkpoint variance or B training (own notes in the slot, bank dropout 0.1) erodes
  the note-following the arm had at 250. 750 (tests running, done ~14:50) and 1000 decide. Page v16.

### 2026-09-18 14:45 — v2B/750: FORCED 6/10, REAL 3/10, offline own 92.7% — 500 was a dip, not a trend
- Forced: one-pick x2, two-pick x2, three-pick green, four-pick green. Failures (end frames in scratchpad frames750/):
  both five-pick scenes pressed the button after three places while the note said "pick up ... for the fourth time"
  (early press, ep 15 at 1259, ep 23 at 897); the four-pick blue scene grabbed the cube on a correct "press" note
  (ep 3 at 504); ep 0 still never lands the first pick. Same shape at 500: after ~3 places the arm presses on its own.
- Real: 3/10 = one-pick x2 + the three-pick green (ep 2, first real three-pick success, with messy notes: "third" written
  right after the first place, one garbage <loc> token, press only after the third place). Failures are writer-side:
  press after 2-3 places in ep 3 / ep 23 (x4 / x5), notes stuck at "second" with the arm hovering in ep 7 / ep 15,
  ep 9 fail, ep 4 fail after a burst of phantom place/pick notes; ep 0 arm.
- B checkpoints so far: forced 6 / 2 / 6, real 3 / 2 / 3, offline own 94.9 / 92.3 / 92.7 (250 / 500 / 750). Page v17.
  1000 tests running (done ~15:10).

### 2026-09-18 14:58 — v2B/1000: FORCED 7/10 (best so far), REAL 2/10, offline own 92.2%
- Forced: the blue three-pick scene (val ep 0) succeeded for the first time in any checkpoint, and the five-pick blue
  did all five places and pressed. Failures: five-pick green (fail during pick 3), four-pick green (timeout at pick 4
  after slow places), three-pick green (timeout after place 2). The arm follows correct notes through five cycles now.
- Real: one-pick scenes only. The writer wrote "press" after one or two real places in 7 of 8 multi-pick scenes; the
  traces show pick/place flipping every tick ("pick2 place pick2 place ...") that inflates the count, and the five-pick
  green looped at "second" for 700 steps. Offline own-writes stays at 92% because it sees demo frames; the closed loop
  sees the robot's own hovering and re-grasps. Writer-side is now clearly the bottleneck.
- B checkpoints: forced 6 / 2 / 6 / 7, real 3 / 2 / 3 / 2, offline own 94.9 / 92.3 / 92.7 / 92.2 (250..1000). Page v18.

### 2026-09-18 17:10 — motion replay tool (user: early press + late release; "visualize the prompt, predicted subtask, view, commit, predicted vs gt 50-step actions in 3D")
- Model: `Pi0.sample_with_memory(fast_decode=True, fast_stop_token=<"|">)` keeps decoding past the sentence terminator
  through the trained FAST branch; the sentence mask, the memory write and the flow suffix see only the sentence
  (training excludes FAST from the suffix view), so notes and expert actions are unchanged. `aux["fast_tokens"/"fast_mask"]`.
- Server: `scripts/serve_yam_memory.py --fast-decode` (Args.fast_decode) returns `fast_actions` [50, 8] (decoded with
  the training FASTSubtaskTokenizer, then the same unnormalize), `fast_ok`, `fast_tokens`, and `prompt_prev_subtask`
  (the "Last:" slot content); `eval/serve_policy.py --fast-decode` passes it through; metadata carries `fast_decode`.
- `eval/panda_fk.py`: numpy Panda forward kinematics (modified DH + hand -45 deg + 0.1034 m tool center); home pose
  check flange (0.307, 0, 0.590) / tcp (0.307, 0, 0.487).
- `eval/demo_replay.py`: LeRobot training episodes (parquet + mp4 via imageio/pyav) through the server at the memory
  tick with own notes; per tick note / dataset label / prompt slot / commit / bank / expert chunk / FAST chunk / the
  demo's next 50 actions, all also as tool-center xyz; front|wrist video; `--random 3 --seed 0`.
- `eval/rollout_replay_export.py`: same format from a recorded rollout (reference = the executed path).
- Page builder (scratchpad `build_replay_page.py`): plotly 3D (rotate/zoom) + gripper panel + prompt/note/label/commit/memory.
- Runs on the eval card 17490595 (watcher stopped 17:08 with the user's OK): `robomme/logs/replay_server.sh 1000 18990`,
  `replay_demos.sh` (3 random demos), `replay_rollouts.sh` (val scenes 3 and 4 own notes, fast logged) -> robomme/replay/v2B_1000/.
- Gripper: action = one continuous value in [-1, 1] (demos strictly +-1: +1 open, -1 closed), state = finger opening 0-4 cm.

### 2026-09-18 17:25 — motion replay results (v2B/1000, page https://claude.ai/artifact/PHHqujtJgpFgQsaDNqcdmQ v3)
- Demos 51 / 63 / 83 (random, seed 0): notes exact 87.2 / 92.9 / 100 %; action expert 2.3 / 1.1 / 1.3 cm from the demo
  path over the 50-step chunk, FAST decode 2.3 / 1.1 / 1.3 cm (FAST is trained and sane; a few ticks decode to zeros).
- Gripper timing on demo frames is NOT late: where the demo releases/closes inside the window, the expert flips within
  ~1 step (release delay -0.2 / -0.3 steps, close -1.9 / -0.9 steps, i.e. slightly early). So the "does not release in
  time, goes up again" behaviour is not an arm timing error under correct context.
- Rollout scene 4 (x2 green, success this run): between the first place and the release the note flipped every tick
  place -> pick 2nd -> ... while the fingers still held the cube (opening 0.018); with a "pick" note in the prompt the
  expert's plan keeps the gripper closed and lifts -> the aborted put-down the user saw. Problem 2 = flipped note.
- Rollout scene 3 (x4 blue, fail): every flip COMMITTED, so at frame 240 the memory held 6 notes (3 places) after ONE
  real place; at 270 the writer, seeing what looks like finished cycles, wrote "press the button to stop" (skipping
  pick 2nd) with the prompt still "place"; the arm followed and pressed at ~345 -> fail. Problem 1 = flip-inflated
  memory -> premature press; the arm itself follows the note.
- Both problems point at the same generic fix: stop tick-to-tick flips from becoming commits (write debounce 2 +
  flip-back retraction 3, both already in the code and off in the bean recipe), and train the writer on the robot's
  own frames. Servers: eval-card server (18990) still up; the pair server (18991) still up, could not be stopped
  (the user's Kerberos caches vanished at ~17:22, no node access).

### 2026-09-18 22:05 — horizon-15 rollouts (plan replaced at every tick), v2B/1000 own notes, replay page v9
- Scene 3 (x4 blue): fail at 286 (h30: fail at 338). Notes: pick 1st -> place (commit) -> label turns to "pick 2nd" at
  210 but the note stays "place" -> at 225 the note jumps to "press" and commits with a CLEAN memory of two notes
  [pick 1st, place]; the arm presses. So the press jump also happens without an inflated bank: the writer, looking at
  the robot's own post-place view, skips "pick 2nd" in a four-times task. Debounce alone will not remove this one.
- Scene 4 (x2 green): timeout at 1302 (h30: success). After the first place the note flipped to "pick 2nd" and
  committed while the fingers still held the cube (0.018 m); prompt "pick 2nd" + holding -> the plan keeps the gripper
  closed -> the view stays consistent with "pick 2nd" -> stuck for 1100 frames (notes 9.2 % exact, 3 commits).
  This is the aborted put-down made permanent: a closed loop between a wrong note and the action it causes.
- Replanning every 15 instead of 30 frames changes nothing about the plan start offset (0.8-2.0 cm either way) and
  does not help; the failures are note-side. Expert-play replays (val 0/1/3/4/15) queued on the same server.

### 2026-09-18 22:15 — held-out expert plays replayed (val 0/1/3/4/15, v2B/1000 own notes), replay page v10; servers stopped
- Notes exact 97.6 / 100 / 84.4 / 100 / 78.9 %; expert plan 1.8-2.5 cm from the planner's path; plan start 1.6-2.0 cm off;
  gripper release/close within ~1-2 steps of the planner (early if anything). Wrong notes: the note stays "place" for one
  tick after the release (val 0/3/15), a burst of the wrong ordinal in the long scenes (val 3 frames 465-510, val 15
  495-555), one "press" before the last place (val 15 f=480), one "place" during the press (val 3 f=570).
- Page data slimmed (raw joint chunks dropped from the embedded data; page 4.2 MB). Both replay servers stopped; the
  eval card and the pair are free. demo_replay.py --records <expert_val> is the expert-play mode.

### 2026-09-19 02:25 — simulator at 30 Hz control (30 steps = 1 simulated second), v2B/1000 own notes, exec 30; replay page v14
- `eval/rollout.py --control-hz 30` injects ManiSkill `sim_config=dict(sim_freq=120, control_freq=30)` through the env
  builder's gym.make (benchmark default 20/100). Manifest carries `control_hz`; the run is marked diagnostic. The
  fps=30 label of videos/datasets was never the control rate: the benchmark runs 20 control steps per simulated second.
- Scenes 3 (x4 blue) and 4 (x2 green): BOTH success at 30 Hz (996 and 567 steps) vs fail / success at 20 Hz. n = 2,
  one seed, a robot the model never trained on (grip closes in 2 steps instead of 1; arm 1.5x faster in sim time), so
  this is a hint, not a result. Notes still flip (exact 62.7 / 63.2 %; scene 3 memory ends with 21 notes for 4 real
  cycles), but the "press" came only after the 4th / 2nd place this time.
- Eval card: the user's placeholder `pi05_base_0919_1h100` stopped with their OK (02:14); replay server (18990) left up.

### 2026-09-19 02:50 — simulator-rate sweeps over all 50 val scenes (v2B/1000, own notes), sweep page
- Page https://claude.ai/artifact/ExDKfTEkoHTQ8qo4NdUc46 (scoreboard + lazy-loaded tick viewer; builder scratchpad
  build_sweep_page.py, data/<key>.json per scene fetched on click). 30 Hz / exec 45 sweep started 02:41 and was stopped
  at the user's request after one scene (val 0: fail at 739, notes 52 %); outputs kept in robomme/replay/sweep_c30_v2B_1000.
- 50 Hz / exec 50 sweep (user 02:47 "sim 50hz and execute 50 action ... all 3 h100"): `rollout.py --control-hz 50
  --execute-horizon 50` (horizon no longer has to be a multiple of the 15-step tick: the 4th chunk of a plan is 5 steps
  and the plan-end query comes 10 steps early). Three fast-decode servers: eval card 18990 (job 17490595) + pair GPU 0/1
  18991/18992 (job 17489557, the user's placeholder stopped with their OK); runners robomme/logs/sweep_sim.sh with
  episode lists 0,3,6.. / 1,4,7.. / 2,5,8..; exports robomme/replay/sweep_c50_v2B_1000/c50h50_epNN.{json,mp4}.

### 2026-09-19 03:15 — 50 Hz sweep, one query per plan (tick = replan = 50 steps), all 50 val scenes, v2B/1000 own notes
- `rollout.py --control-hz 50 --chunk-size 50 --execute-horizon 50 --mode action-chunk-diagnostic` (the diagnostic mode
  lifts the client check "chunk = trained stride"); runner robomme/logs/sweep_sim2.sh (TICK/MODE env); exports
  robomme/replay/sweep_c50t50_v2B_1000/c50h50t50_epNN (the runner's own export missed diagnostic-mode folders, the
  scratchpad export_watch.sh exported them). Page https://claude.ai/artifact/ExDKfTEkoHTQ8qo4NdUc46 v10.
- Result: 13/50 success (fail 33, timeout 4). By pick count: x1 9/10, x2 2/15, x3 2/13, x4 0/6, x5 0/6. Notes right
  79 / 54 / 49 / 38 / 39 %. Same shape as the 15-tick runs: the arm can do the motion, the notes derail after the first
  place. The earlier 50 Hz / 15-tick sweep (stopped at 6 scenes: 2/6, x1 2/2) and the 30 Hz / 45 sweep (1 scene) are
  in robomme/replay/sweep_c50_v2B_1000 and sweep_c30_v2B_1000.
- All three servers stopped 03:16; eval card and the pair (job 17489557) free (their placeholders were stopped with the
  user's OK; the user restarts them if wanted).

### 2026-09-19 03:30 — v3 (user): horizon 40, memory tick 5, longer window, more training on the later pick-ups; 4 x H200
- User 03:30: "on the 4 h200 i want to start v3 training, (1) action horizon set to 40 so predict next 40 actions
  (2) memory tick every 5, so memory write at 4hz, and because we have 4h200 so you can set the window larger and i also
  see it is very poor from second to 3rd or even more, so we can also consider train more on the later pick up, do
  this changes and feel free to run the test and adversarial test and once you done (make sure we take full advantage
  of the h200 ram and cpus) start v3 training enable wandb". Their Qwen PickXtimes training on job 17422727 may be
  killed for it ("you can kill it later"): stopped 03:55 (robomme/logs/kill_qwen_17422727.sh, log .log); the train_hs
  keep-alive (pid 7764) untouched.
- Configs `pi05_robomme_0919_v3A_off` (label writes, memory leaves fresh from base r2/9999, 400 updates) ->
  `pi05_robomme_0919_v3B_off` (own writes from A/400, 1500 updates) + `_v3B_off_smoke` (robomme_config.py, constants
  V3_SEQ_STEPS / V3_A_STEPS / V3_B_STEPS). Same recipe as v2 (official labels, KI base, 16-token prompt slot, no
  visual block, onset CE 3 + first-note confirmation 2 in B) with:
  1. `action_horizon=40`: a plan is 40 steps (2 s at the benchmark's 20 Hz control). FAST tokens of the KI branch
     encode 40 x 8; serving reads the horizon from the config; `rollout.py --execute-horizon` is now checked against
     the served `action_horizon` (was a fixed 50).
  2. `memory_stride_frames=5`: a note decision every 5 frames = 4 per second (v2: every 15 = 1.33 per second). The
     prompt slot ("Last:") carries the label 5 frames back in A and the newest own note in B, as before.
  3. window `memory_seq_steps=160` ticks x 5 = 800 frames = 40 s (v2 covered 600 frames with 40 x 15; median easy /
     medium demo 449 frames, hard 812, so a cold start runs a whole hard demo on its own notes), TBPTT fence
     `memory_block_steps=40` ticks (10 s; v2: 25 ticks of 15 frames), buckets 40/80/120/160; slices leave >= 20 ticks.
     Sized by the 04:27 probe: 120-tick windows at global batch 8 (2 per card) peaked at 102.5 of 144 GB on all
     three bucket shapes (probe_v3B_off_smoke_b8, 3 compiles of 5-7 min, then 22 / 38 / 56 s per update for
     40 / 80 / 120-tick batches); batch 12 would land at ~136 GB (too close for a day-long run), so the headroom went
     into the longer window instead (estimate ~125 GB at 160 ticks).
  4. sampling mix `memory_slice_prob=0.4`, `memory_critical_prob=0.4` -> 36 % cold starts (frame 0, empty memory),
     24 % random slices, 40 % starts within 75 frames before a note change (v2B: 49 / 21 / 30).
  5. NEW generic knob `DataConfig.memory_start_history_power=1.0` (data_loader `_history_power_reweight`): inside the
     slice branch and inside the note-change branch, a window that opens after n label changes of its episode is drawn
     (1 + n) times as often; each branch keeps its total mass, cold starts are untouched. It only counts changes (the
     dataset task column, which agrees with the official sidecar on all 100 episodes) and never reads a sentence. On
     the real data the mass on starts after >= 4 changes is 25.6 % (cold 36 %, after 1 change 12 %, 2: 8 %, 3: 5.5 %,
     4: 7 %, 5-9: 5 -> 2 %). Unit tests src/openpi/training/data_loader_v3_history_test.py (4, pass); config_test,
     v5_generic_test, data_loader_v6/v35 tests pass (a first version of the validation line swallowed the rest of the
     generic-task checks; caught by v5_generic_test, fixed).
- Adversarial data check on the real dataset (robomme/logs/check_v3_data.py, log check_v3_data_*.log, PASS): items are
  [120, 40, 32] actions / [120, 32] state; the FAST branch of every checked step decodes back to 40 x 8 actions with
  mean abs error 0.007-0.010 (normalized units); the 5th target of chunk k matches the state of step k+1 better than
  the 15th (0.050 vs 0.124; the next-action offset holds at stride 5); step 0 of a cold start says "Last:none", step 1
  carries step 0's label; sidecar vs task-column change counts agree 100/100 (min/med/max 3/7/11).
- Launcher: robomme/logs/chain_v3_h200.sh (A then B, W&B project robomme_memory, JOB 17422727, GRES 4, fsdp 4,
  CPUS 32, --num-workers 16) via chain_v3_ctl.sh start|status|stop on iris-hgx-2; batch from the probe
  robomme/logs/probe_v3_h200.sh (6-update run with preallocation off, nvidia-smi peak per card in probe_v3_status.log).
- **Launched 04:30** (chain_v3_ctl.sh start on iris-hgx-2: BATCH=8, WORKERS=16, MEMFRAC=0.95, W&B on): stage A
  `robomme_0919_v3A_off` (400 updates, checkpoints 200/400) then stage B `robomme_0919_v3B_off` (1500 updates,
  checkpoints every 250, 500-multiples kept). Expected ~57 s per update with 160-tick windows -> A ~6.5 h, B ~24 h,
  done around 09-20 midday; job 17422727 ends 09-21 18:58. Logs robomme/logs/train_robomme_0919_v3{A,B}_off.log
  (W&B may freeze that file: live log = v35/wandb/wandb/run-*/files/output.log), chain_v3_status.log.
- 05:25 check: stage A at update 42 (first update 04:40 after a 5.5 min compile; four window shapes compiled by
  update ~10), 19-72 s per update by bucket (tqdm average 35-54 s), GPU utilisation 96-98 % on all four cards, no
  out-of-memory across the 160-tick batches (17 % of batches). W&B run oajjfp4f
  (https://wandb.ai/kewalk-stanford-university/robomme_memory). Projection: A done ~09:40, B done ~05:00 on 09-20.
- **09:42 (user: "Run eval")**: stage A at update 376 (ce 0.67 at 200, 0.43 at 300; ~49 s per update). The v3 eval
  battery runs on the eval card (job 17490595, GPU 0) through the checkpoint watcher robomme/logs/watch_v3.sh
  (watch_v3_ctl.sh start|status|stop, started 09:44): for every v3A checkpoint the offline LABEL-writes test, the
  offline OWN-writes test, the forced rollout and the real rollout; for every v3B checkpoint own / real / forced.
  Runners offline_eval_v3.sh, rollout_v3_h100.sh, rollout_v3_forced.sh = the v2 ones with `--chunk-size 5` (the client
  requires the tick to equal the trained stride) and execute horizon 5 (memory update and plan update together, as the
  user asked for v2's h15 runs); output dirs offline_<exp>_<step>[_oracle] and rollout_<exp>_<step>_h5[_forced]; ports
  18960+ (A) / 18980+ (B). The user's note "250A is ok for switching" could not apply: A saves at 200 and 400 only and
  was 22 min from 400, so B starts from A/400 (~10:06).
- 09:52 A/200 offline LABEL-writes test (6 expert val plays, tick 5): notes right 97.2 % (phase 97.6 %), press 5/5,
  early flips 18 over ~3x the queries of v2 (v2A/250: 96.8 / 97.5, 8 early flips). Stage A writer healthy at tick 5.
- 09:58 stage A DONE (exit 0; ce 0.30 at 400, checkpoints 200/400). **Stage B started 09:59:45** from A/400 (chain, W&B).
  A/200 offline OWN-writes test: 85.3 % exact (phase 96.1 %), press 5/5, 18 early flips; the long plays (x4 78 %, x5 71 %)
  pull the mean down, x1/x2 at 99-100 % (v2A/250 own: 92.2 % at a third of the ticks). Forced rollout of A/200 running
  from 09:57 (tick 5, execute 5).
- **10:03 (user: "I think I have 2h100 available use that to run rollouts")**: the single watcher loop was replaced by
  three shared-queue workers (robomme/logs/v3_worker.sh via v3_workers_ctl.sh start|status|stop, run on iris-hgx-1):
  eval card (job 17490595, ports 18900+), pair GPU 0 and GPU 1 (job 17489557 via srun --overlap, ports 19000+/19100+).
  A worker waits until its card is free, then claims the next leg by creating its output directory (atomic on NFS):
  per checkpoint v3A real, forced, own, label; v3B real, own, forced. The watcher's running forced-A/200 leg was left
  to finish on the eval card. First assignments: pair0 real A/400, pair1 real A/200. Per-worker JAX caches
  (robomme/cache/eval_jax_<worker>) so three compiling servers never share one cache file.
- Viewer page v20 (https://claude.ai/artifact/KFwEp9Pxz8kZhy7jAQus5R): v3 rows appear as scenes land (hz 5 filter for
  v3 runs in build_clean_page.py; RUN table in build_rollouts_section.py; rebuild_clean_page.sh includes 0919_v3).
- **10:10 (user: "Can we do this execute 20 action and replan")**: extra legs `real20` / `forced20` (rollout dirs
  `rollout_<exp>_<step>_h20[_forced]`): the model is still queried every 5 steps (memory tick = trained stride, the
  client insists), a new plan is adopted every 20 steps (4 ticks per plan; h5 = every tick). Worker v2
  (robomme/logs/v3_worker2.sh; order v3A real, real20, forced, forced20, own, label; v3B real, real20, own, forced,
  forced20) replaces v3_worker.sh through v3_handover.sh, which kills an old loop only between legs (no running
  rollout lost) and starts the new worker for that card. Page: v3 h20 rows show as runs "v3A20"/"v3B20".
- 10:10 stage B first update (ce 0.30 at update 0; all 159 leaves matched from A/400; 5.6 min compile). W&B run
  x8htphrs (https://wandb.ai/kewalk-stanford-university/robomme_memory/runs/x8htphrs); 1500 updates -> ~06:30 on 09-20.
- **10:13 (user: "Stop current and just run this [execute 20 and replan]")** after the h5 diagnosis: in the h5 rollouts
  the arm HOVERS (A/400 real scene 1: after step 150 the arm sits above the cube with the right note until the 1301-step
  cap; A/200 forced scene 4: holds the cube above the table for 1100 steps). frames/predictions of that scene: mean joint
  movement 0.037 rad/step in the first 100 steps, 0.009 afterwards; a plan's step 1 / 5 / 10 / 20 / 40 sits 0.04 / 0.06 /
  0.08 / 0.29 / 0.80 rad (sum over joints) from the current pose, so executing only the first 5 steps and replanning
  re-anchors at a near-static start every tick and the arm never gets going. All h5 legs killed (partial dirs kept:
  forced A/200 1/8 before the stop, real A/200 1/3, real A/400 0/1). Workers now run h20 only (v3_worker2.sh: v3A real20,
  forced20, own, label; v3B real20, own, forced20; ctl ports 18950/19050/19150).
- 10:18 first execute-20 scenes: the arm moves normally again. A/200 real h20: scenes 1 and 5 success in 247 / 246
  steps, scene 4 fail (307); A/400 real h20 scene 1 success; A/200 forced h20 scene 1 success. (v2B/1000 x1 successes
  took ~270-300 steps at exec 30.) Page v23 rows "v3A20" (plan kept 20 steps).
- 10:30 execute-20 tallies: A/200 forced 6/10 (x1 2/2, x2 2/2, x3 1/2, x4 0/2 (scene 7 timeout), x5 1/2), A/200 real
  3/8 so far, A/400 real 2/10 (x1 only), A/400 forced 2/2 so far. Page v26. Stage A at 400 is worse than at 200 with its
  own notes; the B stage decides. (The SUCCESS n/m line in each leg's .out counts every scene started after the leg's
  T0 across all parallel legs; use the page / manifests per run.)
- 10:33 stage B after its compiles (5 compiles, updates 0-7): 38 / 56 / 56 / 75 s per update by window length, ~55 s
  average -> B/250 ~13:50, B/1500 ~09:30 on 09-20 (job ends 09-21 18:58).
- **10:37 stage A execute-20 battery complete** (page v27): A/200 forced 6/10, real 3/10 (x1 2/2, x2 1/2, rest 0),
  own-notes test 85.3 %, label test 97.2 %; A/400 forced 3/10, real 2/10 (x1 only), own-notes test 87.0 % (press EARLY
  on the x5 play; 24 early flips), label test running. So the extra 200 A updates helped the offline note score a
  little and hurt the arm under forced notes (6/10 -> 3/10). Workers idle until B/250 (~13:50).
- 13:38 B/250 landed (~13:33; B ce 0.23 at 100, 0.18 at 200; 52.6 s/update, B/1500 ~08:00 on 09-20). Legs running: forced20
  (eval card), real20 (pair 0), own-notes test (pair 1). First forced scenes 2/3 (x1 1/1, x2 1/2). A/400 label test 97.0 %.
  Page v28.
- 13:44 B/250 own-notes test 89.2 % (phase 96.9 %, press early on the x5 play, 25 early flips; A/400 87.0, A/200 85.3).
  Forced h20 2/6 so far (timeouts on x2 scene 9, x3 scenes 0 and 2), real h20 2/4 (x1 both). Page v29.
- **13:55 B/250 complete: forced h20 3/10, real h20 2/10, own-notes 89.2 %.** The forced timeouts are the SAME stall as
  h5, one level up: B/250 forced scene 9 (x2) after the note "pick up ... second time" the arm hovers above the cube
  from step 260 to 1301 (speed 0.012 rad/step vs 0.035-0.05 while working); scene 7 (x4) holds the cube over the target
  with the "place" note for 1100 steps. Plan geometry in the stall: |plan step 20 - now| 0.14 / 0.28 rad but
  |step 40 - now| 0.50 / 0.74 rad, gripper command 0.84 -> -0.44 (scene 9: descend and close in the second half) and
  -0.99 -> 0.97 (scene 7: release at the end). The plans intend the motion in their SECOND half; executing 20 re-anchors
  in the near-static first half (v2 ran 30 of 50 and got through). Worker v3 (v3_worker3.sh, ctl restarted 13:57) adds
  `real40` / `forced40` legs (whole plan executed) for the B checkpoints; B/250 runs them now. Own-notes scene 0 (x3)
  does not stall: it fails on notes (flips + a few raw "Action: <loc...>" decodes as the note).
- 13:58 own-notes detail, B/250 real scene 0 (x3): 5 of 261 decodes came out as raw "Action: <loc...>" text; none entered
  the bank (the first-note confirmation / debounce filtered them; memory_v7_write_vocab_only stays off). The real failure
  is counting ahead: the bank read "first, second, third pick" by step 291 before any place, then alternated
  "third pick / place" for the rest of the episode. Page rows for the whole-plan runs: "v3B40".
- 14:05 whole-plan (h40) B/250 so far: forced 3/8, real 3/5 (page v32) -- no timeouts any more, the arm moves. The
  remaining forced failures are semantic, by the benchmark's rules (robomme_env/PickXtimes.py: during a pick/place phase
  a fail = a non-target cube picked up or the button pressed; during the press phase a fail = any cube picked up):
  scene 4 (x2) got both cycles right, then with the forced "press the button" note went back to the cube on the target
  and grasped it (fail at 557 in the press phase); scene 3 (x4) did three cycles right and failed at 585 in the fourth
  pick phase (hovering low over the cube on the target; button press or wrong cube by the rules). Same reading as v2's
  press diagnostic: the arm does not follow the "press" note against the picture.
- **14:07 B/250 whole-plan complete: forced 3/10 (x1 2/2, x2 1/2, rest 0), real 3/10 (x1 2/2, x2 1/2)**; no timeouts
  except real scene 0 (x3). Versus v2B/250 (exec 30 of 50, tick 15): forced 6/10, real 3/10. Page v33. Next: B/500 ~17:20.
- **17:11 (user: "ok i think we have 500 now you can use all 3h100 to test")**: B/500 had landed (~16:50; ce 0.12 at
  400, 0.13 at 500) but the workers were waiting: the user's placeholder trainings `pi05_base_0919_1h100` / `_2h100`
  (started ~16:20) held 74-76 GB on all three cards. Stopped 17:13 on the user's word; workers claim the B/500 legs.
- 17:21 B/500 own-notes test: MEAN exact 89.8%  phase 97.5%  press ok 5/5  early flips 15 (B/250: 89.2 %). First execute-20 scenes: forced 3/3, real 2/2 (page v34).
- **17:38 B/500 complete (page v36)**: own-notes test 89.8 %; execute 20: forced 5/10 (x1 2/2, x2 2/2, x4 1/2; timeouts
  0, 2, 7, 23; fail 15), real 3/10 (x1 2/2, x2 1/2); whole plan: forced 3/10 (x1 2/2, x4 scene 3), real 2/10 (x1).
  B/250 was forced 3/10 / real 2/10 at execute 20 and 3/10 / 3/10 whole-plan. Next B/750 ~20:30.
- **20:05 (user: "ok i think 750 is on the way, and use this one to eval: kewalk · job 17490595")**: the pair 17489557
  goes back to the user (their Qwen3-VL processes started there 20:03); our pair workers stopped 20:10. One eval worker
  (v3_worker4.sh, `PAIR=0 GPU_FREE_MAX=20000`) stays on the eval card 17490595 and treats it as free below 20 GB used,
  because the user's small Qwen/benchmark processes share that card. B/750 (~20:10) gets the five legs one after another
  on that card (~1.5 h per checkpoint).
- 20:25 B/750 forced execute-20: 4/10 (x1 2/2, x2 1/2, x4 scene 3; six timeouts = the hover stall). Real execute-20 running
  (2/2 so far). Page v38. B/750's own-notes and real legs had been claimed by the pair workers seconds before those were
  stopped (20:07); the orphaned claim dirs were removed so the eval worker re-runs them.
- **20:45 (user: "add a mode using correct notes as current subtask, but on the video also show the model's own
  prediction")**: forced mode + own note. `serve_policy.py --show-own-prediction` (with --diagnostic-oracle-subtask):
  at every tick the server first decodes the model freely on the same observation and memory state (result discarded:
  no write, no state change), returns it as `own_subtask` / `own_confidence` (the least certain token's probability),
  then runs the forced step as before (the true sentence goes into the bank, the slot and the arm). `rollout.py`
  writes both into frames.jsonl / predictions.jsonl; `recording.py` draws the forced sentence on one line and
  "model's own note: ... (xx%)" beneath it (amber when it differs, green when it matches). Metadata
  `own_prediction_shown`. rollout_v3_forced.sh passes the flag, so every forced leg from B/750's whole-plan leg on
  carries it (the earlier forced runs do not). Cost: two model calls per tick in forced runs.
- 20:47 B/750 real execute-20: **4/10** (x1 2/2, x2 2/2 in 998 / 1055 steps; the other six timed out = hover stall) --
  equals v2B/1000's best real score (4/10 at exec 30) and is the first v3 checkpoint to clear both two-pick scenes on
  its own notes. Forced execute-20 4/10. Own-notes test + whole-plan legs running. Page v39.
- 21:17 own-note overlay LIVE (B/750 whole-plan forced leg, metadata own_prediction_shown=true): scene 0 (x3) the
  model's own note differs from the forced one on 55 of 156 ticks (e.g. step 285 truth "pick 2nd", own "place"). B/750
  whole-plan: real 4/10 (x1 2/2, x2 2/2), forced 4/4 so far; own-notes test 89.1 %. Page v40. The B/750 forced
  execute-20 leg (recorded before the change) is re-queued so it also carries the overlay (old dir *_pre_ownnote).
- **21:52 B/750 complete**: own-notes test 89.1 %; execute 20: forced 4/10 (first run) / re-run with the own-note overlay
  see page, real 4/10 (x1 2/2, x2 2/2); whole plan: **forced 6/10** (x1 2/2, x2 2/2, x3 scene 0, x5 scene 23; best forced
  so far), **real 4/10** (x1 2/2, x3 scene 2, x4 scene 3; the two x2 scenes failed). B at update 890, ~46 s/update;
  B/1000 ~23:20, B/1500 ~05:50.
- 23:45 B/1000 first two legs: own-notes test **87.8 %** (down from 89.1 at 750; press ok 3/5, 20 early flips); real
  execute-20 **2/10** (x1 2/2 in 299 / 264 steps; both x2 scenes fail early at 297 / 418 steps; x3/x4/x5: three
  timeouts, two fails). Worse than B/750 real (4/10). Forced execute-20 running (2/2 so far), whole-plan legs queued.
  Page v43. B at update 1040, 53 s/update: B/1250 ~02:50, B/1500 ~06:35.
- 00:05 (09-20) B/1000 forced execute-20: **4/10** (x1 2/2, x2 2/2 in 1014 / 636 steps; x3 scene 3 and x4 scene 7 fail
  at 634 / 890, four timeouts). Same forced score as B/750's execute-20 leg. Whole-plan legs running. Page v44.
- 00:16 B/1000 whole-plan real: **3/10** (x1 2/2, x3 scene 3 in 695 steps; the two x2 scenes fail at 249 / 538, five more
  fails at 245-651 steps, one timeout). With the whole plan executed the stall is gone but the model fails on its own
  notes instead. Forced whole-plan leg running (last B/1000 leg). Page v45.
- **00:20 (09-20, user: "lets stop training on 4h200 now")**: v3 chain stopped on iris-hgx-2 (chain pid 3383283 + train.py
  3493511, SIGTERM, clean exit; the 1 GB train_hs keep-alive 7764 untouched; GPUs back to ~1 GB). Stage B ended at
  update ~1090 of 1500; last checkpoint B/1000 (500/750/1000 kept, 250 deleted by keep_period). Final v3 table so far:
  A/200 own-notes 85.3 % forced-20 6/10 real-20 3/10; A/400 87.0 / 3 / 2; B/250 89.2 / 3 / 2 (whole plan 3 / 3);
  B/500 89.8 / 5 / 3 (3 / 2); B/750 89.1 / 3 (re-run) / 4 (6 / 4); B/1000 87.8 / 4 / 2 (forced whole plan running / 3).
- **00:33 B/1000 complete** (page v46): own-notes test 87.8 %; execute 20: real 2/10, forced 4/10; whole plan: real 3/10,
  **forced 5/10** (x1 2/2, x2 2/2, x4 scene 7; five fails, no timeout). Best v3 checkpoint stays B/750 (real 4/10 at
  both horizons, forced whole plan 6/10). Training stopped at 1090 (user), so this is the last v3 checkpoint.
- **00:50 (09-20, user: "remove all previous diagnose stuff ... keep basic fast loss, lm loss, flow loss ... log loss every
  10 steps")**: W&B slimmed. `TrainConfig.log_diagnostics` (new, default True): False drops every `diagnostic/...` and
  per-position metric from W&B and the console, keeping `loss`, `flow_loss`, `ce_loss`, the new `lm_loss` (sentence
  tokens) and `fast_loss` (FAST action tokens; both telemetry-only token-weighted splits of the same CE from the v32
  sequence loss: `ce_lm` / `ce_fast`), the side losses, `grad_norm` / `memory_grad_norm` / `param_norm`, the sequence
  stats and the new `lr` (the optimizer's schedule at the logged update). All three robomme config builders now set
  `log_interval=10, log_diagnostics=False` (the v3B run logged 83 keys every 100 updates; now 14 keys every 10). The
  diagnostics are still computed inside the step (nothing removed from the model), only not logged. Note: the full-tree
  grad/param norms are sampled once per log window, so now every 10 updates (cheap next to a 50 s step).
  `pi0_v32_test::test_end_to_end_sequence_ce_reaches_queries_and_slow_memory` fails on a pre-existing
  `KeyError: 'decision_mask'` (the v6.3 decision weight, before the new lines run); the other tests pass.
- 01:07 logging change tests: `pi0_v32_test` 9/9 (the pre-existing decision_mask failure deselected), `pi0_v5_test`
  26/26, `pi0_v6_test` 14/14 on CPU (the three files together segfault on iris-ws-18 when run in one pytest process;
  one file per process is fine). User 01:05: "you can use the single h100, kill the placeholder there" -> their
  `pi05_base_0920_1h100` (pid 2510702) stopped, 3-update smoke `robomme/logs/smoke_logging.sh` (v2B smoke config,
  `--log-interval 1`) started on the eval card at 01:07 to see the new log line. User 01:06: after the tests/run,
  occupy all H100s with the placeholder (`openpi_trossen/cluster_scripts/train_pi05_base.sh`); the pair 17489557 already
  runs the user's own `pi05_base_*_2h100` (75 GB per card), so only 17490595 gets one after the smoke.
- 01:16 logging smoke on the single H100: OOM (a 42 GB allocation at the first step), identical byte count to the same
  v2B smoke config's OOM on 09-18 04:59, i.e. the memory smoke config does not fit one 80 GB card regardless of the
  change (the CLI cannot shrink the window because the loader requires the last bucket to equal memory_seq_steps).
  GPU check of the new keys therefore waits for the first real multi-GPU launch (first log line at update 10). The
  CPU tests (v32/v5/v6, 49 tests) cover the new v32 outputs. Placeholder started on 17490595 at 01:17 per the user
  ("after you finish all test / run tell gpu placeholder to occupy all h100s"): `train_pi05_base.sh 17490595 1 8 16 8
  1h100`; the pair 17489557 already carries the user's own 2h100 placeholder.
- 01:55 (user: "show me the detail of our structure model ... great visualization"): two reference pages published.
  Recipe page https://claude.ai/artifact/1MSAFUkvwx6kUm3RFhnQd4 (one tick, token layout, bank, read/write, slot, training
  windows, serving, all numbers) and model-anatomy page https://claude.ai/artifact/MrEv8BRw4TFEMLGgwWX2Hp (two experts in
  the same 18 blocks, one block's insides, attention visibility grid, memory interface, parameter counts, tensor shapes,
  plain-words description). Facts resolved by instantiating pi05_robomme_0919_v3B_off; parameter counts from
  nnx.eval_shape: total 3.498 B, trainable 2.495 B (Gemma-2B blocks 1,982 M, embedder 527 M frozen, action expert 425 M,
  SigLIP 415 M frozen, note-bank path 83 M, old visual memory 62 M frozen, projections/norms 5 M).
- 15:06 (09-20) **B/750 label-writes offline test: 97.2 %** exact (phase 97.7; ep0 95.9, ep1 100, ep3 94.8, ep4 98.8, ep15 96.5),
  identical to A/400 (97.0) -> stage B did not erode the read of a correct bank on expert frames; the 69 % steady-state
  agreement measured on the robot's own correct-notes rollouts is entirely the trajectory shift (slow / hovering arm).
  First attempt at 14:55 OOM'd: a `pi05_base_0920_1h100` placeholder step started on the eval card 37 s after ours was
  stopped; rerun under `label_chain_0920.sh` with a 10 GB holder on the card. B/1000 skipped (user: one video only).
  Overlay video of ep15 (five picks): `offline_robomme_0919_v3B_off_750_oracle/labelwrites_B750_ep15.mp4`. Placeholder
  restarted on 17490595 at 15:06.
