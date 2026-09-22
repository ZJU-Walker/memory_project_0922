# memory_project_0920 — 0920_v0 (branch v0920)

Clean copy of the RoboMME tree (memory_project_robomme @ ab09cac + its uncommitted working state) for the **0920_v0**
method (user decisions 2026-09-20; full plan in `robomme/docs/0920_v0_plan.md`). Shared, read-only data comes in by
symlink (`data`, `v5`, `v6`, `robomme/data`, `robomme/expert_val`); checkpoints, logs, caches and W&B runs are this tree's own.

## What 0920_v0 changes (versus v3, pi05_robomme_0919_v3B_off)

| | v3 | 0920_v0 |
|---|---|---|
| start weights | KI task base r2/9999, then stage A (label writes) -> B (own writes) | the ORIGINAL pi0.5 base; one run, own writes from update 0, label-write probability 1 -> 0 over the first 3k of 10k updates |
| read | 8 queries = learned bank + instruction shift + previous-note shift + cross-attention conditioner, injected after block 8, pointer bonus on the decode | 8 fixed learned queries, tokens appended to the INPUT (visible to all 18 blocks), no conditioner, no pointer bonus |
| write | on change, confidence >= 0.9, debounce | every valid tick, no rule (same token-level whitened keys, linear delta-rule bank, decay 0.99/tick) |
| prompt | `Task, State, Last: <slot>` (112 tokens) | `Task, State` only (80 tokens) |
| images per tick | front + wrist + a blank third slot (3 x 256) | front + wrist (2 x 256) + 4 past FRONT frames at t-0.8/-0.6/-0.4/-0.2 s, 2x2-pooled to 64 tokens each + a learned time embedding per slot; all history masked for 20 % of training samples |
| tick / window | 5 frames / 160 ticks (40 s), horizon 40 | 20 frames (1 s) / 40 ticks (40 s), horizon 40, buckets (10, 20, 30, 40), TBPTT block 20 |
| W&B | robomme_memory (diagnostics) | project `robomme_0920`: loss, flow, lm, fast, lr every 10 updates |

## Code map (all under `openpi/`)

- `src/openpi/models/pi0_config.py` — `memory_v0920_input_read / _input_rms / _history_frames / _history_pool / _history_dropout` (+ validation).
- `src/openpi/models/pi0.py` — `_v0920_input_scale`, `v0920_read_tokens`, `_v0920_prepare_prefix` (one full-depth pass over
  [prefix | 8 tokens]); `_v32_layer_mask` early = late under the flag; history pooling + time embedding in `embed_prefix`;
  `_top_camera_token_count`; per-step image masks + history dropout in the sequence scan (`xs["image_valid"]`); history frames
  share the front camera's augmentation; label-write draw (`label_u` / `label_write_prob`) in the write decision;
  `sample_with_memory` preprocesses whatever image keys the request carries.
- `src/openpi/models/model.py` — `Observation.seq_label_write_prob`; `scripts/train.py` fills it from `TrainConfig.label_write_schedule_steps`.
- `src/openpi/training/config.py` — `DataConfig.memory_image_history_frames / _stride / _key`, the repack keys `observation/history_<i>` + `observation/history_valid`;
  `src/openpi/training/data_loader.py` — interleaved front-camera `delta_timestamps` + `SplitImageHistory`;
  `src/openpi/transforms.py` — `SplitImageHistory`, history stacking in `BuildMemorySequence`.
- `src/openpi/policies/robomme_policy.py` — `RobommeInputs(history_frames, drop_blank_camera)`: `history_<i>_rgb` image keys with their masks, blank right wrist dropped.
- `src/openpi/training/robomme_0920_config.py` — `pi05_robomme_0920_v0` (+ `_smoke`), registered from `config.py`.
- serving: `scripts/serve_yam_memory.py` publishes `image_history_frames/stride`; `cluster_robomme/eval/common.py` client ring
  buffer (`observe()` on every frame, `_with_history` builds `observation/history_*`), decoder allowlist; `rollout.py` observes when either history is trained.
- tests (CPU): `src/openpi/models/pi0_v0920_test.py` (8), `src/openpi/training/robomme_0920_test.py` (3).
- BinFill (09-21): `cluster_robomme/generate_robomme_demos.py` (extra expert demos with the benchmark recorder, GPU-pinned workers),
  `cluster_robomme/prepare_robomme_h5_to_lerobot.py` (released + generated H5 -> one lossless LeRobot dataset with the lowercased
  official labels, manifest / subtasks_official / prepared_official / norm_stats under the dataset name), `robomme_config._load_spec(task=)`,
  `robomme_0920_config.binfill_configs` (`pi05_robomme_0920_binfill_v0` + `_smoke` + `_probe`; dataset from `OPENPI_0920_BINFILL_DATASET`,
  default BinFill200), launch `robomme/logs/train_binfill_ctl.sh` (runner `train_binfill_h100.sh`, 2 x H100 job 17489557).
- launch: `robomme/logs/train_0920_ctl.sh smoke|start|stop|status` (runs `train_0920_v0_h200.sh` on the node; `cluster_robomme/run_train.sh` points at this tree).

## Log

- 2026-09-20 22:20 — worktree + venv, model/config/data/serving changes, 11 CPU tests green, configs instantiate
  (repack carries the 4 history keys, prefix images = base + 4 history + wrist, weight loader = original pi05_base). Next: loader
  probe on the node, `_smoke` on the 4 x H200 17422727 (user 22:11: "for gpu test you can use this"), then the run.
- 2026-09-20 22:53 — loader probe OK (6 image keys, history masks, prompt 64/80); `_smoke` OK on 17422727 (pi05_base audited:
  51 matched / 93 fresh / 0 ignored; step 0 loss 36.6 = lm 23.3 + fast 17.9 + flow 0.55, finite grads; 3 steps + checkpoint,
  exit 0). Two tree fixes: the training identity check refuses paths outside the tree, so `v35/cache/.../pi05_base/params` and
  `robomme/data` are hard links (cp -al) into this tree instead of symlinks. REAL RUN started 22:53: `pi05_robomme_0920_v0` /
  `robomme_0920_v0`, batch 8 on the 4 x H200 of 17422727 (job ends ~19:00 09-21; the launcher resumes from the last
  1000-step checkpoint on the new job: `train_0920_ctl.sh start` on that node with JOB=<id>).
- 2026-09-20 23:47 — batch probe on the 4 x H200: batch 8 = 10-12 s/update at 96 % util, batch 32 = 15-30 s (20 s mean), i.e.
  2.2x the windows per hour. User: "yes lets do batchsize 32 with 5k steps" -> config now batch 32, 5k updates, cosine to 5k,
  label-write ramp 1.5k, checkpoints every 500 (keep 1000s), 24 workers; ~28 h. Batch-8 run (47 updates) and the probe were
  stopped and their empty experiment dirs removed; the real run `robomme_0920_v0` restarted 23:47 (W&B robomme_0920).
- 2026-09-21 01:04 — REVIEW + RELAUNCH. A 7-dimension code review (agents; 32 verifier agents lost to an API outage, the
  critical finding re-verified by hand) found one training bug: under the v0 flags the visual bank's write tokens are zeros,
  which the delta rule never "commits", and the CE + flow loss of every DECISION tick (count sentences, button press) was
  still gated on that commit -> those ticks had zero loss since the start (lm_loss 0.04 by step 120 was the give-away).
  Fixed in pi0.py (a valid transition counts; state valid from tick 0 under v0) + regression test. Also in this relaunch:
  the frozen image tower now runs once for all ticks before the scan (`memory_v0920_vision_outside_scan`, H100 pair A/B:
  12.1 vs 15.0 s/step, -20 %, losses track), the analytic label prefill decays notes as write-every-tick would
  (`MemorySequenceSubtasks.write_every_step`), label-write probability shape robust to grad accumulation, W&B drops the
  v3.5 side / v5 separation placeholders (weights 1e-6), launcher defaults batch 32 / 24 workers. Serving fixes (no
  restart needed): `_sample_with_memory_v32` preprocesses the request's own image keys (the earlier patch sat in the
  legacy `sample_with_memory` body), `V5SentenceMemory(write_every_step=...)` re-commits every tick like training, the
  client ring buffer no longer stores frame 0 twice. Run 1 (buggy, 152 updates) log kept as
  `train_robomme_0920_v0_run1_buggy.log`; exp dir recreated; run 2 started 01:04 (same config, batch 32, 5k updates).
- 2026-09-21 01:18 — review follow-ups: model/data history-frame cross-check (`LeRobotRobommeDataConfig.create`), hold-smoke
  guard in rollout.py, history keys in the v3.5 compaction, remat-policy knob (`OPENPI_0920_REMAT`, default nothing_saveable).
  Tests now 13 (`pi0_v0920_test.py`: + sampler without the blank camera, history dropout, augmentation sharing, hoisted-tower
  equivalence, decision-tick gating) + 4 (`robomme_0920_test.py`: + prefill gap shift), all green on CPU. Perf probes on the
  2 x H100 pair (`robomme/logs/probe_chain_h100_0921.sh`): remat dots_saveable / everything_saveable / XLA latency-hiding
  flags vs the hoisted baseline 12.1 s per update; results in train_probe_h100_*.log. Candidates for the resume only:
  finer buckets (10,15,...,40), a faster remat policy if it fits in memory. Not for this run: causal buffer 208 -> 128,
  sentence_len 48 -> 16, bf16 vocab head, batch 64, lower FSDP degree.
- 2026-09-21 01:45 — H100 probe chain done (batch 4, fsdp 2, updates 10->30): baseline 15.14 s, hoist 12.27 s, hoist +
  remat `dots_saveable` 11.22 s (loss@30 13.3 vs 14.1, numerics differ slightly), `everything_saveable` OOM (50 GB alloc),
  XLA latency-hiding flags crash the compiler (abort in PyClient::Compile). Decision: `dots_saveable` at the RESUME only,
  via `robomme/logs/train_0920_v0_h200_v2.sh` (tries it first, falls back to nothing_saveable on an OOM exit); the ctl
  script now launches v2. Run 2 untouched (step 77 at 01:44, ~20 s/update mean incl. compiles).
- 2026-09-21 03:20 — BinFill on the 2 x H100 (user 02:24 "start training the binfill task using exactly same setup? make full use
  of that 2 h100"; 02:47 "before you train, you can use the Robomme provided tool to generate another 100 training demos";
  02:55 "adjust the window length and our sample rate, if this task is shorter we can include more full eps"; the 4 x H200 run
  stays untouched). Data facts (released BinFill, 100 eps, 60,282 frames): 266-1044 frames per episode, median 620 — LONGER
  than PickXtimes (266-1025, median 543); 15 sentences ("pick up the <first..fourth> <color> cube", "put it into the bin",
  "press the button", "All tasks completed"), 36 goals, 8 segments per episode (4-12), shortest segment 32 frames. The
  PickXtimes window (40 ticks x 20 frames = 800) would cut 23/100 episodes -> BinFill uses tick 25 frames (1.25 s) x 42
  ticks = 1050 frames: every episode fits one window for ~5 % more sequence (tick 20 would need 53 ticks, +33 %); block 21,
  buckets (12,22,32,42), critical pad 125; everything else exactly V0 (horizon 40, 4 history frames 0.2 s apart, every-tick
  writes, label ramp 1.5k of 5k updates, prompt 60-65 of 80 tokens). Extra demos: `generate_robomme_demos.py` drives the
  benchmark's own recipe (`tests/_shared/dataset_generation.py::_run_one_episode`: gym.make + RobommeRecordWrapper + fail-aware
  planner over task_list, seed + attempt retries, recovery on local episodes 0-5) with 8 workers on the two H100s; seeds
  24000 + 100 i (released train 4000-13902, val 1.04 M, test 540 k), difficulty easy,easy,medium,hard repeating like the release.
  Gotcha: the recorder buffers HDF5 frames ONLY inside its video branch (`_video_should_record` = save_video and ...), so
  save_video=False "succeeds" with 0 frames (first attempt, 100 empty files, archived as gen/BinFill_novideo_empty); fixed,
  relaunched 03:14 (~11 episodes/min with videos, ~5 GB GPU per card, H5 on node-local /scr/kewalk/robomme_0920/gen/BinFill).
  Conversion: `prepare_robomme_h5_to_lerobot.py` (released + extra -> BinFill200; BinFill100 = released only for probes,
  prepared 03:2x). Configs `pi05_robomme_0920_binfill_v0(_smoke|_probe)`; tests 4 + 2 green. Chain `binfill_after_gen.sh`:
  BinFill200 conversion + batch probes 8 / 12 / 16 on the pair (BinFill100 spec, dots_saveable), then the real launch via
  `train_binfill_ctl.sh start` with the largest fitting batch.
- 2026-09-21 03:45 — BinFill data done: generation 100/100 in 12 min (57,978 frames, 257-1148 per episode, mean 1.13 seed
  attempts, 37 GB H5 + 1.4 GB overlay videos on /scr/kewalk/robomme_0920/gen/BinFill); BinFill200 = released 100 + generated
  100 -> 118,260 frames, the same 15 sentences, max 12 segments per episode (`robomme/metadata/BinFill200`,
  `robomme/data/lerobot/BinFill200`, `robomme/assets/BinFill200`); 4 of the 200 episodes exceed the 1050-frame window
  (kept at 42 ticks). Config test against BinFill200 green. First H100 probe: batch 8 + dots_saveable at 42 ticks OOMs
  (60.8 GB allocation, train_probe_binfill_b8.log) -> second chain `robomme/logs/probe_binfill_h100_b.sh`: batch 8 and 6
  with the default remat, then batch 4 + dots; the first fitting setting goes into `probe_binfill_fit.txt` and the launch.
- 2026-09-21 04:10 — BinFill REAL RUN launched: `pi05_robomme_0920_binfill_v0` / exp `robomme_0920_binfill_v0` on the 2 x H100
  17489557 (iris-hgx-1), batch 6, fsdp 2, remat nothing_saveable, 16 workers, W&B robomme_0920, dataset BinFill200, 5k updates,
  label ramp 1.5k, checkpoints every 500. Probe results at 42 ticks on 80 GB cards: batch 8 OOMs with dots_saveable (60.8 GB
  alloc) and with nothing_saveable (54.6 GB); batch 6 + nothing_saveable fits at 16.9 s/update (21 updates in 355 s) -> ~23.5 h
  for 5k updates, before the job ends 09-23 21:12. Control: `robomme/logs/train_binfill_ctl.sh status|stop`, runner
  `train_binfill_h100.sh` (`REMAT=nothing_saveable BATCH=6 WORKERS=16 ... start`), logs train_robomme_0920_binfill_v0.log.
- 2026-09-21 13:10 — **HANDOFF: `robomme/docs/HANDOFF_20260921.md`** (written for a session taking over with no
  context: job ownership, access, both live runs and their resume commands, the BinFill pipeline, the gotchas, disk,
  and what comes next). Time-critical item in it: the 4 x H200 job 17422727 ends 18:58 today with PickXtimes near
  step 3400 of 5000, so resume on the new 4 x H200 with `JOB=<id> bash robomme/logs/train_0920_ctl.sh start`.
- 2026-09-21 13:55 — PickXtimes VAL SWEEP of 0920_v0 update 2000 on the user's free 1 x H100 (job 17533970; user 13:33 "run the eval on
  pickxtimes ... on all 50 and pub to artifact"): `robomme/logs/eval_0920_val50.sh 2000` (lanes = `eval_0920_lane.sh`: own model
  server + simulator loop, shared episode queue via mkdir claims) as an --overlap step of 17533970; tick 20, execute 20 of the 40
  planned actions, official 20 Hz, own notes, seed 7; exports in `robomme/replay/val50_0920_v0_2000/` (json + 768-px mp4 per
  episode, summary.json at the end). Two lanes OOMed (each server peaks ~40 GB in the write step; 78 GB used) -> one lane,
  ~1 episode/min, GPU 13-19 %. Page https://claude.ai/artifact/12ZbLxDKQi993fnS8KFPce (builder scratchpad build_val50_page.py,
  incremental republish via val50_publish_list.py). First 3: 0/3, notes right 76 / 71 / 21 %.
- 2026-09-21 14:10 — Val sweep of 0920_v0/2000 STOPPED after 7 episodes (user 13:57 "the low level pick up skill is so bad ...
  stop eval now and lets analysis"): 0/7 success, the simulator's label left "pick ... first time" in only 1/7 (v3B/750 at the same
  execute horizon 20: 9/10 first picks on overlapping scenes, 4/10 successes). Grasp analysis (`robomme/replay/val50_0920_v0_2000`
  exports vs the v2B_1000 expert plays of scenes 0/1/3/4): the arm closes at the right TIME (steps 70-91 vs expert 75-95) and at
  table height, but every close but one is on air (finger width 0); tool-center distance to the cube at the first close 6 / 22 /
  24 / 7 cm; in scenes 1 and 3 the arm went to the place target / another cube instead of the instructed cube (wrist view shows
  the ring). Plans are followed (plan-vs-executed 2-3 cm), so the plans are wrong, not the tracking. Notes are a second failure:
  they advance pick -> place -> pick 2 -> press -> done on a clock while the label never moves. Flow loss 0.0055 at 2000 vs v3B
  0.0029 (v3 had ~17k task-specific updates before its memory stage; v0 starts from the generic pi05 base). Page updated (7 eps).
- 2026-09-21 14:25 — HISTORY-MASK DIAGNOSTIC (user 14:13 "can you do this?"): the same checkpoint 2000 on scenes 0-6 with every history
  slot marked invalid at serving (`rollout.py --mask-history`, client `HttpPolicyClient.mask_history`; exports
  `robomme/replay/val50_0920_v0_2000_nohist`, lane knobs TAG/EXTRA_ARGS/EPS in eval_0920_lane.sh). Result: the arm does NOT improve --
  0/7 picks (1/7 with history), first close 6.0 / 14.4 / 23.2 / 8.0 cm from the cube (history on: 6.0 / 22.1 / 24.2 / 6.9), closes on
  air as before. So the history frames are not what breaks the arm. Side effect: notes drift less without history (mean notes-right
  65 % vs 37 %; 3/7 episodes never left "pick first", which is correct because nothing was picked). Page updated with both variants.
- 2026-09-21 14:45 — v1 CONFIGS = v0 WITHOUT THE PAST FRAMES (user 14:32 "remove history image so still 2 images"): new
  `pi05_robomme_0920_v1` / `_v1_smoke` and `pi05_robomme_0920_binfill_v1` / `_smoke` (robomme_0920_config.py `_history_overrides`,
  `history=0` on `_v0` / `_v0_task`): model `memory_v0920_history_frames=0` (dropout 0), data `memory_image_history_frames=0`; the
  RoboMME input mapping falls back to the v3 layout (front + wrist, blank third slot masked and never attended, no history keys),
  everything else bit-identical to v0 (test `test_v1_is_v0_with_the_two_current_cameras_only` compares every model / data / train
  field; 7/7 green on hgx-1). v0 / binfill_v0 untouched (run 2 resumes on them). One real loader batch of v1_smoke on the node (CPU):
  images base/left_wrist/right_wrist [32, 20, 224, 224, 3], right_wrist mask 0.000, no history keys, prompt <= 64 of 80 tokens;
  `config_0920_check.py` prompt probe fixed (needs an actions field), `common.py` mask_history read via getattr (test client has no
  __init__). Serving needs nothing: `image_history_frames` 0 in the served metadata -> the client sends no history. NOT launched
  (no launcher for v1 yet; train_0920_v0_h200_v2.sh is live for run 2 and must not be edited -- copy it when the user decides).
  Discussion item pending with the user: is there a short-term history the model should keep once the frames are gone.
- 2026-09-21 15:05 — ONE-FRAME UP/DOWN CHECK (user 14:56: without visual history, inside "pick up the ... cube" can the model tell
  going down from going up?). `robomme/logs/pick_updown_from_one_frame.py` on PickXtimes_official (100 eps, CPU): 24,438 pick-labelled
  frames, 22,098 moving > 1 cm over the next tick. Gripper width alone: closed (1.83 cm = cube held) -> up 86 %, open (4.0 cm) -> down
  90 %. Tool xyz (FK of the joint state) + width, leave-one-episode-out 7-NN: direction right 97.8 %. The 2.2 % wrong sit at the two
  turning points (grasp instant at the table, 1.4 % of those frames; top of the lift at ~12 cm with the cube held, where the transport
  to the ring starts within the next second), i.e. "when does the next phase start" within one tick, never "which phase am I in".
  Label switch pick -> place: 267/267 with the gripper closed, 28-44 frames after closure, tool 12.9-14 cm above the table -> one frame
  (cube held, lifted) determines the switch. Conclusion: the frame + 8-D state carry the phase; no visual history needed for this.
- 2026-09-21 15:58 — v1 LAUNCHED ON THE 4 x H200 (user 15:08: "make action horizon prediction to 30 instead of 40 and use 4h200 start
  training ... bigger batch ... you can stop current training, make sure this v1 start from plain base pi05"; 15:41: "just try batch
  size 64 if not working use batch size 32"). v1 redefined = v0 without the past frames AND without the blank third camera slot
  (new model flag `memory_v0920_drop_blank_camera`, read by LeRobotRobommeDataConfig.create -> RobommeInputs drop_blank_camera; the
  slot never took part in attention, so this only removes 256 padding tokens per tick: prefix 600 vs 856 tokens), action horizon 30,
  batch knob OPENPI_0920_V1_BATCH; BinFill v1 gets the same (horizon 30, two cameras). Tests 20/20 (config + model), loader batch on
  the node: images base/left_wrist only, actions [32, 20, 30, 32]. Run 2 (pi05_robomme_0920_v0, 4 x H200) STOPPED 15:16 at update
  ~3560 (checkpoints 1000/2000/2500/3500 kept under robomme/checkpoints/pi05_robomme_0920_v0/robomme_0920_v0). Launcher
  `train_0920_v1_h200.sh` + `train_0920_v1_ctl.sh status|smoke|probe|start|stop` (BATCH + BATCH_FALLBACK, OOM -> next batch, fresh
  dir each attempt). Attempt batch 64: RESOURCE_EXHAUSTED at the first step (142.9 GB temp allocation per card vs the 136.5 GB pool) ->
  batch 32 launched 15:57 (exp robomme_0920_v1, W&B robomme_0920, mode=fresh, loader = openpi-assets pi05_base). Sizing from that
  failure: ~8.9 GB of activations per window at 40 ticks (~223 MB per window-tick, 0.70x v0), so 48 (12/card, ~107 + ~20 GB fixed)
  would fit with ~9 GB headroom and 56 would not; the user chose 32 as the fallback. Job 17422727 ends 18:58 -> resume v1 on the next
  4 x H200 with `JOB=<id> BATCH=32 bash robomme/logs/train_0920_v1_ctl.sh start` (run_train.sh resumes when the exp dir has a
  numeric checkpoint). Tools kept: aot_memory.py / aot_memory_h200.sh (ahead-of-time memory analysis, unused), dump_batch_spec.py.
- 2026-09-21 23:30 — BEAN SCOOP WITH THE 0920 v1 STRUCTURE, PREPARED (user 22:46: "apply current setup to our real world task the
  bean scoop task ... adapt to yam station ... start from pretrained pi05 scoop base, you can use 2h100 ... for ticks lets use the
  current bean scoop setup, only change the structure here ... inspect first and confirm with me the training config"; 22:59 "2h100
  is working now, so prepare your code first and once my work finished you can start training"). New module
  `openpi/src/openpi/training/beans_0920_config.py` -> `pi05_yam_beans_0920_v1` (+ `_smoke`), registered at the end of config.py:
  `_v7_boba_mem_variant` (the boba2B recipe) with model_overrides = the v1 STRUCTURE (V0_MODEL minus horizon / token length / window /
  state mask / prefill / history keys; drop_blank_camera False because the station has three real cameras; oracle False, prev
  committed, own content re-applied after the variant) on boba_0913_v2 with the two-phase labels v3 + manifest v2 (copied from v7
  into openpi/cluster_v7/boba, sha256 pinned), boba base 9999 with fresh memory leaves, tick 15 / 60 ticks / block 30 / buckets
  (20, 40, 60) / slice 0.9 / critical 0.5 / pad 75 / prefill 26 / horizon 50 / max_token_len 80 / state mask 0 / RTC delay 15 /
  lr 5e-5 (warmup 100, constant) / AdamW clip 1 / memory grad clip 5; 3000 updates, label-write ramp 900, save 250 / keep 500 /
  newest 2, batch 4 fsdp 2 workers 12, W&B project beans_0920, checkpoints under beans/checkpoints (run_train.sh gained LOGS /
  CKBASE env overrides). Test `beans_0920_test.py` (structure == pi05_robomme_0920_v1, everything else == boba2B) green; 0920
  tests 7/7. Launcher `beans/logs/train_beans_v1_h100.sh` + `train_beans_v1_ctl.sh status|smoke|start|stop` (BATCH 4, fallback 2,
  refuses to start while any process holds > 2 GB). NOT launched: the 2 x H100 is the user's until they say so, and the config
  awaits their confirmation. Context: BinFill v0 on the same pair ended 16:24 with exit 143 (SIGTERM, not ours; checkpoints 1000 /
  2000 / 3000 / 3500 / 4000 kept); RoboMME v1 ran to the job end 18:58 with checkpoint 500 saved; a new 4 x H200 (17425063, hgx-2)
  started 18:59 for the user. Question of a new folder: a worktree needs the 0920 work committed on v0920 first (22 files).
- 2026-09-22 00:20 — beans0922: THE REAL-ROBOT LINE IN ITS OWN WORKTREE (user 09-21 23:34 "no not the boba task, it is the led bean
  scoop task"; 23:42 "if you cannot find it lets just train a brand new one"; 23:52 "train a base policy first ... knowledge
  insulation, use 2h100 job 17489557, train up to 10k steps, and once it is ready ... train our current setup ... on bean task,
  make sure you create a clean folder ... train it till 5k steps ... ref the old bean v5 training"; 23:55 "you can always use the
  2h100 now"; 23:56 "commit and push to my github ... make everything portable"). Findings: the pi05 KI bean base
  (`pi05_beans0905_base_v7rtc_20260906_r1/{5000,10000,15000}`), every v5 memory checkpoint AND the LeRobot dataset
  `v5/data/lerobot/yam/bean_scoop_0905_v5` are gone (no copy anywhere; no record); the 89 raw demos, labels, manifest and norm
  stats survive. Done: 0920 work committed on v0920 (4a96349); worktree `memory_project_beans0922` (branch beans0922) with
  data/v5/v6 links, hard-linked pi05_base, own venv; `beans0922_config.py` = `pi05_yam_beans0922_base` (09-06 KI recipe, 10k,
  batch 16 fsdp 2, public pi05_base) + `pi05_yam_beans0922_v1` (0920 v1 structure on the beans v5/B9 window: tick 5 frames, 40
  ticks, TBPTT 25, buckets 14/27/40, prefill 16, state mask 0.5, horizon 50, RTC 15, lr 2.5e-5, from base/10000 with fresh memory
  leaves, 5k updates, label ramp 1500, batch 4 fallback 2); portable launcher `beans/logs/train_beans0922.sh` (root-relative,
  srun only when JOB is set, env overrides OPENPI_BEANS_*), chain `chain_beans0922.sh`, `beans0922_ctl.sh`, `beans/README.md`.
  Dataset rebuild `convert_beans0905.sh` running as an srun step (plain-ssh python on hgx-1 lands in the tiny interactive-job
  cgroup and crawls -- the earlier stalls); `start_after_convert.sh` runs the base smoke and starts the chain when it is done.
- 2026-09-22 00:50 — user 00:36: label-write ramp over the FIRST 500 updates ("at 500 it is already fully self"), NO state masking
  (memory_state_mask_prob 0), launch, and push to the new repo. Config updated (ramp 500, state mask 0.0), duplicate-keyword bug in
  `memory_config` fixed (STRUCTURE already carries prefill_history), `openpi/cluster_robomme/eval` un-ignored and added (the generic
  `eval/` ignore had kept the RoboMME client code out of the 0920 commit). Pushed: `beans0922` -> `main` of
  github.com/ZJU-Walker/memory_project_0922 (remote `origin0922`, ssh key id_ed25519). Chain auto-starts after the dataset rebuild.

---

# memory_project

Workspace for fine-tuning Physical Intelligence's **π₀.₅ (pi05)** on the **bimanual YAM** `bin_memory_banana` task.

## Layout

```
memory_project/
├── data/bin_memory_banana/   # raw bimanual YAM teleop demos (30 episodes, ~30 Hz)
├── openpi/                    # cloned openpi repo + venv (gitignored), with YAM data config & policy transforms
└── i2rt/                      # cloned i2rt repo (YAM URDF, FK, MuJoCo SimRobot)
```

YAM-specific code lives under `openpi/`:

- `openpi/examples/yam/convert_yam_data_to_lerobot.py` — dataset converter.
- `openpi/src/openpi/policies/yam_policy.py` — input/output transforms (`YamInputs`, `YamOutputs`).
- `openpi/src/openpi/training/config.py` — data config (`LeRobotYamDataConfig`) and train config (`pi05_yam`).
- `openpi/src/openpi/models/pi0_memory.py` + the `pi05_yam_memory` config — the episodic-memory variant (see [Memory-as-Context](#memory-as-context-episodic-neural-memory) below).

## Data format

Each `data/bin_memory_banana/demoN/` folder contains, at ~30 Hz:

- `{left,right}_joint_positions.npy` — `(T, 7)` follower state (6 arm joints + 1 gripper per arm).
- `{left,right}_control.npy` — `(T, 7)` leader/teleop command (the action target).
- `top_camera_rgb.mp4`, `left_camera_rgb.mp4`, `right_camera_rgb.mp4` — 3 cameras (640×480).
- `metadata.json`.

The converter builds a standard **LeRobot** dataset:

- `state` = `concat(left_joint_positions, right_joint_positions)` → **14-dim** (actual follower state).
- `actions` = `concat(left_control, right_control)` → **14-dim** (leader command).
- `image` / `left_wrist_image` / `right_wrist_image` = top / left / right cameras.

The `pi05_yam` config trains with **delta actions** on the 6 arm joints of each arm and keeps the **grippers absolute** (mask `[6 delta, 1 abs, 6 delta, 1 abs]`), which handles the leader/follower gap when the gripper grasps an object. There is no language instruction in the data, so a fixed prompt `"find the bin with banana"` is injected.

## Setup

Activate the openpi venv before running anything:

```bash
source /iris/u/kewalk/memory_project/openpi/.venv/bin/activate
```


## Pi05 steps

### 1. Convert raw demos → LeRobot dataset

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run examples/yam/convert_yam_data_to_lerobot.py \
    --data_dir /iris/u/kewalk/memory_project/data/bin_memory_banana
```

Writes the dataset to `$HF_LEROBOT_HOME/yam/bin_memory_banana`. Append `--push_to_hub` to also publish it to the Hugging Face Hub.

### 2. Compute normalization stats

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam
```

### 3. Train (fine-tune π₀.₅ from `pi05_base`)

```bash
cd /iris/u/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_yam --exp-name=yam_banana_pi05 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_memory.py pi05_yam_memory \
    --exp-name=yam_banana_pi05_mem_400m_v3 --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_yam/yam_banana_pi05/`. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` so JAX can use up to 90% of GPU memory.

### 4. Serve the trained policy for inference

WebSocket inference server on port 8000:

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_yam \
    --policy.dir=/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam/yam_banana_pi05/6000

# v1 — 3-camera memory encoding, forced via mem_camera=all:
uv run scripts/serve_policy_memory_v1.py \
--policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v1/<step>

# v2 — top-camera-only (training config default, nothing forced):
uv run scripts/serve_policy_memory_v2.py \
--policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v2/<step>
```

`<step>` is the checkpoint iteration to load (e.g. `29999` for the final step of a 30k-step run). Run inside `tmux` so it survives terminal disconnects.


## Memory-as-Context (episodic neural memory)

A MAC/Titans-style **online episodic memory** on top of `pi05` for the hidden-bin task: during one
episode the robot opens both bins (sees the banana), closes the lids, then must open the *remembered*
bin. A small memory MLP whose weights update **online within the episode** stores the banana location
during inspection and is read back at recall; its read/write projections are trained end-to-end by
the action loss. Everything is **additive** — the `pi05_yam` baseline above is left untouched.

New pieces (baseline `Pi0` / `pi05_yam` / `scripts/train.py` unchanged):

- `openpi/src/openpi/models/pi0_memory.py` — `Pi0Memory`: memory module + the causal split-cost
  unroll (write-only memory frames that skip the LLM, then one full action forward at the target).
- `openpi/src/openpi/models/pi0_config.py` — `Pi0MemoryConfig` (`d_mem`, `mem_eta/theta/alpha`,
  `freeze_vision`).
- `openpi/src/openpi/training/config.py` — `LeRobotYamMemoryDataConfig` + the `pi05_yam_memory` `TrainConfig`.
- `openpi/src/openpi/training/memory_data_loader.py` — episode-sequence loader (N causal frames + 1 action chunk).
- `openpi/scripts/train_memory.py` — trainer (reuses the baseline `train_step`; bakes in a cuBLAS GEMM flag).

### 1. Compute normalization stats

Same transforms as `pi05_yam`, so the stats are identical (already copied into
`openpi/assets/pi05_yam_memory/`). To recompute:

```bash
cd /iris/u/kewalk/memory_project/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam_memory
```

### 2. Train (joint fine-tune `pi05` + memory from `pi05_base`)

```bash
cd /iris/u/kewalk/memory_project/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train_memory.py pi05_yam_memory --exp-name=yam_banana_memory --overwrite
```

Checkpoints land in `openpi/checkpoints/pi05_yam_memory/yam_banana_memory/` — **saved every 1000
steps, kept permanently every 5000** (same as `pi05_yam`). The SigLIP vision tower is frozen; the
Gemma LLM + action expert + memory module train jointly. The first compile of the `n_mem=16` episode
unroll takes a few minutes (cached afterwards under `$JAX_COMPILATION_CACHE_DIR`).

**Key knob — `mem_theta`** (the online memory learning rate, default `1e-2`): too small (e.g. the
original `1e-5`) and the memory barely moves within an episode and stores nothing. Sweep `~[1e-3, 1e-1]`
and watch `||M - M_0||` grow without blowing up; add `mem_alpha > 0` (forgetting) if it drifts.
`n_mem` (default 16) is the number of causal frames the memory unrolls over per item.

### Validation (smoke-tested 2026-06-30, single H200, from `pi05_base`)

```text
# weight loader keeps the 12 memory params; the rest load from pi05_base
memory keys in ref: 12 | present in loaded: 12
loss: 1.15  finite: True
nonzero memory grads: 12/12          # every memory param learns; frozen vision gets 0 grad

# data loader yields obs [B, N, 224, 224, 3] + a single action chunk [B, H, 32]
actions: (2, 50, 32) | image base_0_rgb: (2, 16, 224, 224, 3) | state: (2, 16, 32)

# jitted train_step on real batches (n_mem=4 for a fast smoke)
step 1: loss=0.0945 grad_norm=2.6329 param_norm=1803.75
step 2: loss=0.1106 grad_norm=3.3861 param_norm=1803.75
step 3: loss=0.0698 grad_norm=2.7867 param_norm=1803.75
```
