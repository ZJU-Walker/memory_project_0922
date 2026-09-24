# beans0922 — LED bean-scoop memory policy (real YAM station)

Current ablations use template-slot SNAP, per-row A250→B3000: global batch 16 on 4 H200, 12 on 4 H100, both ACCUM=1.
See [the current runbook](../README.md) and [mechanism](ablations/README.md). The sections below preserve historical runs,
including token-bank v4e, and are not launch instructions for the new ablations.

The real-robot line of the 0920 memory structure: a pi0.5 that writes its own sub-task sentence into a small fast-weight bank
every tick and reads it back through 8 learned queries at the input (see `openpi/src/openpi/training/robomme_0920_config.py`
for the structure and `beans0922_config.py` for this task). Task: "scoop the beans into the tray as many times as the green
light blinked" — the count must be remembered from the LED blinks at the start of the episode.

## What runs

| step | config | from | length | cards |
| --- | --- | --- | --- | --- |
| 1 | `pi05_yam_beans0922_base` | public `pi05_base` | 10k updates, batch 16 | 2 GPUs, FSDP 2 |
| 2 | `pi05_yam_beans0922_v1` | step 1's `beans0922_base/10000` | 5k updates, batch 4, label-write ramp 500 | 2 GPUs, FSDP 2 |

Step 1 is the plain knowledge-insulation base (sub-task sentence + FAST tokens supervise the language side, flow matching
trains the action expert under a stop-gradient prefix). Step 2 adds the memory: tick 5 frames (0.17 s, so every LED blink is
seen), 40-tick windows, TBPTT 25, own writes every tick, label content with probability 1 -> 0 over the first 500 updates (fully self-written from 500 on), no state masking.

## Setup on a new machine

```bash
git clone <this repo> memory_project_beans0922 && cd memory_project_beans0922/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen            # python 3.11 venv at openpi/.venv
```

Data and weights (all paths are relative to the repo root; override any of them with the environment variables in the table):

| what | default location | override |
| --- | --- | --- |
| LeRobot dataset `yam/bean_scoop_0905_v5` (89 episodes, 71,089 frames, 53 GB) | `v5/data/lerobot/yam/bean_scoop_0905_v5` | `OPENPI_BEANS_DATASET_ROOT` |
| norm stats | `v5/assets/pi05_yam_bean_scoop_0905_v5/yam/bean_scoop_0905_v5/norm_stats.json` | `OPENPI_BEANS_ASSETS_DIR` (dir above `yam/`) |
| pi05 base weights for step 1 | `gs://openpi-assets/checkpoints/pi05_base/params` (downloaded on first use) | `OPENPI_BEANS_PI05_BASE` |
| step-1 checkpoint for step 2 | `beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params` | `OPENPI_BEANS_BASE_PARAMS` |
| labels + manifest (in the repo) | `openpi/cluster_v5/beans/beans_v5_subtask_labels_0905_v7tgt.json`, `beans_episode_manifest_0905_v1.json` | — |

To rebuild the dataset from the raw demos (`data/0905beans_{1,2,3}`, 89 demo folders with the three camera mp4s, joint
streams and the LED/go signals): `HF_LEROBOT_HOME=<root>/v5/data/lerobot python examples/yam/convert_yam_data_to_lerobot.py
--episode-manifest data/0905beans_episode_manifest_v1.json --repo-name yam/bean_scoop_0905_v5` (about 25 min on a CPU node).
`MEMORY_PROJECT_ROOT` overrides the repo root if the tree is not where the scripts live.

## Launch

```bash
# this cluster (Slurm, inside an allocation): the whole chain, detached
JOB=<job id> GPUS=0,1 bash beans/logs/beans0922_ctl.sh start
# another machine where you already own the GPUs: no JOB -> python runs directly
GPUS=0,1 bash beans/logs/beans0922_ctl.sh start
bash beans/logs/beans0922_ctl.sh status | stop     # smoke_base / smoke_mem / base / mem run one stage
```

The launcher refuses to start while another process holds more than 2 GB on the chosen cards, resumes an experiment dir
that already holds a numeric checkpoint, and in step 2 retries once at batch 2 if batch 4 runs out of memory. Logs:
`beans/logs/train_<exp>.log`, status lines in `beans/logs/train_beans0922_status.log`. W&B project `beans0922`.

## Throughput note (measured 2026-09-22 on iris-hgx-1)

The loader memory-maps the dataset's arrow index (the `datasets` cache) and decodes the mp4s per sample. With both on a
network filesystem the two H100s sat idle 80 % of the time (about 2 s per update at batch 16). With both on the node's
local disk the same run does 1.8 updates/s. On a shared cluster set, before launching:

```bash
export OPENPI_BEANS_DATASET_ROOT=/scr/<user>/beans0922/bean_scoop_0905_v5   # rsync -a of the dataset dir
# the arrow cache: the launcher sources cluster_v35/env.sh, which pins HF_DATASETS_CACHE to <repo>/v35/cache/huggingface/datasets
# (exporting the variable yourself is overridden), so make that in-tree directory a symlink to the local disk:
mkdir -p /scr/<user>/beans0922/hf_datasets && ln -sfn /scr/<user>/beans0922/hf_datasets <repo>/v35/cache/huggingface/datasets
```

The cache is rebuilt there on the first loader start (a few minutes from a local dataset). `beans/logs/switch_at_5000.sh` shows the
restart used here (resume from the last checkpoint). Symptom to recognise: loader workers at ~5 % CPU in `folio_wait_bit_common`
(page-fault waits on the memory-mapped arrow files) while the GPUs idle.

## Serving

`openpi/scripts/serve_yam_memory.py` serves step 2 with the same tick, write rule and read as training (the RoboMME clients
in `openpi/cluster_robomme/eval` show the request format; the YAM robot client for the LED task is
`openpi/examples/yam/client_memory_v5_led.py` from the v5 line and needs the 0920 request fields ported before the robot test).

## Ablations

`beans/ablations/` holds the ablation rows of this policy (same recipe on 4 cards, 3000 updates, label ramp 500, from the
base 10k): the control row and (1) "snap + visual memory" (`Pi0Config.memory_vis_bank`). One-time setup on another machine
= `beans/ablations/setup_other_cluster.sh` (clone, venv, dataset + base checkpoint from the Hub); one row =
`beans/ablations/run_<row>.sh`. See `beans/ablations/README.md`.

## Offline held-out probe (videos)

`beans/eval/run_heldout_videos.sh <config> <exp> <step>` walks the six development episodes (manifest split
"development": LeRobot indices 25 29 59 64 72 73, never trained on) at the training tick with the note bank carried across
ticks, decodes the subtask sentence every tick, writes it back (`self`: own sentences every tick, as deployed; `oracle`: the
label sentences) and renders the top camera with the labelled phase, the decoded sentence and the bank overlaid
(`openpi/scripts/v5_heldout_video.py`, which dispatches to the 0920 input-read prefix for these models). Output:
`beans/eval/videos_<exp>_<step>/ep<idx>_<mode>.{mp4,json}` + `status.log`. On this cluster run it inside a Slurm job you own:
`JOB=<job> GPU=<card> GRES=<cards of that job> bash beans/eval/run_heldout_videos.sh pi05_yam_beans0922_v1 beans0922_v1 1500`
(one H100/H200, ~2 min per episode and mode). Elsewhere: `bash beans/eval/run_heldout_videos.sh ...` on a node with a free card.
A third pass with the bank never written: `MODES=self TAGSUF=_blank EXTRA='--intervention blank' ...` (files
`ep<idx>_self_blank.*`). Results page for checkpoint 1500 (2026-09-22): https://claude.ai/artifact/WM4QNWYaVRDAQyEiG4R6rR

Checkpoint-1500 verdict (6 dev episodes): the running blink count is read from the bank (own notes exact on 113/127 light
ticks, right count in the notes in 5/6; with the bank blank both blinks decode as "light on: 2"), but the "yellow go: scoop k
times" count ignores the notes -- own notes and label notes give the same sentence in 6/6 (right in 3/6), and the every-tick
own writes then drift 2 -> 3. Hence `pi05_yam_beans0922_v2` = v1 with change-only, confidence-gated (0.9) writes
(`V2_WRITE_RULE` in `beans0922_config.py`).

**v3 (approved 09-22 15:25) = v2 + "look before you ask" + error-driven token weight.** `pi05_yam_beans0922_v3`
(`V3_QUERY_CONTEXT`): (1) writes only on change with confidence >= 0.9 (v2); (2) `memory_v0920_query_context=True` -- each of
the 8 learned read questions is shifted, before it is asked, by its own attention over the tick's image + prompt tokens and by
the mean embedding of the last committed note, both through zero-initialised maps (fixed questions at init; the answers still
enter at the input for every block; new leaves `memory_sem_query_context_pooler/_context_proj/_prev_proj`, fresh-init by the
loader); (3) `memory_v7_hard_token_ce_weight=5.0` -- sentence tokens the model's own teacher-forced prediction gets wrong at
that tick weigh 5x (the count word is 1 token in ~50; the weight fades once learned; switches on the `v7_hard_token_count`
telemetry). Tests: `openpi/src/openpi/models/pi0_v0920_query_context_test.py`, `beans0922_test.py`. Gates at every 500 updates:
`scripts/v5_count_flip_eval.py` (true-note accuracy and flip-follow >= 0.9, blank ~1/3) + the held-out video probe (own = label
go count, right in >= 5/6). Stop rule: flip-follow < 0.5 at 2000 -> v4 = v3 + B9's slot table instead of more training.

**v3 verdict and the measurement behind v4 (2026-09-23, `beans/eval/token_bank_geometry.py`, `v3_question_shift_probe.py`,
`v3_query_kernel_norms.py`; JSON results next to them).** Checkpoint v3/1000 videos: the first go count was right in 4/5
readable episodes but drifted within the go phase, and "scoop k of x" was near random. The probes replay an episode's true
notes into a fresh bank and read it three ways. (1) The store is fine: a stored digit reads back at cosine 1.00 with its exact
context key ("light off:", "scoop 1 of"), and the three digit values sit at cosine 0.75-0.88 from each other. (2) v3's
question shift was 170x longer than the base questions (the note embedding carries Gemma's sqrt(width) scale) and both shift
maps were rank one, so all 8 questions collapsed into one: the 8 answers were identical to three decimals and the answers for a
2-blink and a 3-blink episode were 0.92-0.98 alike through the scoop phase (0.63-0.77 with the fixed questions alone). Do not use
`memory_v0920_query_context` as it stands. (3) An answer is a blend over the whole history: the go note fades 1 % per tick
(exact-key read strength 1.00 -> 0.59 -> 0.36 -> 0.21 -> 0.15 across four scoop notes 40 ticks apart, cosine to the true digit
0.71 at "done") while the newest note dominates, so "of x" was a noisy copy that flipped and then stayed wrong; the mean
probability gate never fired at a wrong count. Also measured: the sentence-only count battery on v3/1000 answers the go count
right with a BLANK history (63/63 first-go steps), so on the six development episodes the go count is not a memory read at all
(appearance); that battery is not a memory test for the go step here.

**v4 "token bank, exact copies" (user 09-23 23:55 "ok开始做"; keeps the token bank).** `pi05_yam_beans0922_v4`
(`V4_TOKEN_EXACT` + `V4_BANK` in `beans0922_config.py`): v2's change-only write rule and the 5x token weight, no question shift,
plus (a) `memory_v6_pointer_read` in context mode (beta 10, now admitted under the input read): while a token is decoded the
bank is asked "what followed this exact context last time" and the answer is added to the token scores, so "scoop 2 of _"
fetches the previous x exactly and "scoop _" the previous k; (b) `memory_v0920_prev_readback`: the last committed note is read
back through the bank with its own write keys and enters the input as one exact token per note position (48 tokens after the
8 questions; new leaves `memory_sem_readback_inject_w`, `memory_sem_readback_slot_embedding`), so the tray decision sees k and
x instead of a blend; (c) bank decay 0.01 -> 0.001 per tick on both banks (half-life 115 s); (d) `memory_v5_write_conf_min`:
the gate compares the lowest token probability with 0.8 (trainer, video script and robot server alike). Still learned rather
than exact: the yellow-go count and the first scoop note's x (new contexts) come from the 8 fixed questions reading the fresh
light-off / go note. Tests: `openpi/src/openpi/models/pi0_v0920_v4_token_test.py`, `beans0922_test.py`. Launcher
`beans/logs/train_beans0922_v4.sh` (2xH200 batch 8; on 4xH100 batch 8 = 2 per card, fallback 4). Step-0 CE is ~250 (fresh
memory leaves; v1 started at 90, v3 at 38) and is under 10 by step 30. Gates as before: videos at 1000, then the sentence
battery (its go-step number is appearance-driven here; read the scoop-phase videos).

**v4 own-note probes and v4b (2026-09-23 05:25).** Own-note rollouts of v4 at 500 and 750 (pages linked from the 250 page):
the trained gate (lowest word probability 0.8) let one note per episode through, because the model's own light-phase
sentences decode at 0.4 to 0.5; with 0.5 applied offline the copy chain was exact and confident at 500 but the go count was
still guessed (no light note ever written), and by 750 the model guessed confidently and overrode its own notes (27 writes
flickering "of 2" / "of 3" on episode 25). Label-note rollouts at 500 were right in 6/6 episodes, so the read design holds and
the gate was the fault. Teacher-forced decision/evidence exact stayed 0.99/0.92 throughout, because most training windows
start with the label history prefilled: judge these models by the own-note videos, not by those curves. `pi05_yam_beans0922_v4b`
(`V4B_WRITE_RULE`) = v4 with the gate at 0.3 plus the two-tick confirmation (`memory_v7_write_debounce_steps=2`), the same rule
in training, in `v5_heldout_video.py` (self mode) and in `serve_yam_memory.py`; resumed from v4's checkpoint 500 (moved to the
v4b experiment directory) on the two H200s, launcher `beans/logs/train_beans0922_v4.sh` with `CFG=pi05_yam_beans0922_v4b
EXP=beans0922_v4b`. Note (ablation session, 05:35): in the training scan the two-tick confirmation also applies to label notes,
so under v4b a label note enters the bank at the second tick of its sentence (one tick later than v4); every beans sentence
lasts several ticks at the 5-frame tick and the window prefill writes label notes directly, so the ramp changes by one tick
of delay per label note.

**v4b own-note rollouts at 750 and 1000, the own-content bias, and v4c (2026-09-23 08:53; pages
`X3czMsuo7hLqgyW3NQXU52` (750) and `95shkGGZgvQWzQWwXjD2Ab` (1000)).** The gate fix worked: at 750 the light notes were written
and mostly right (episodes 25, 29, 73 exact; 64 wobbled; 72 one "off 3"; 59 missed blinks). The yellow-go count did not follow
them: with the last light note saying 3 the model opened with "scoop 3 times" (episodes 64, 72; training episodes 0 and 3),
with 2 it said 2 (25) or pushed to 3 (72, label notes: "light off: 3" at the onset tick), and with 1 it said 2 in every case,
held-out and training alike (29, 73, 59, training episode 8), at 0.94-0.96 confidence, with a clean bank ("wait", "on 1",
"off 1") in front of it. Picture memorisation is ruled out (training episodes fail the same way) and so is a broken read
path (checkpoint 500 with label notes answered 1 for both x=1 episodes; the injection gates sit at their 0.5 init in both
checkpoints, the pointer beta at 10). Label-note rollout of episode 29 at 750 (bank written from the labels: "wait", "on 1", "off 1") also opened with "scoop 2 times" at 0.96, so the bias does not depend on who wrote the bank. The explanation that fits the asymmetry: from step 500 the bank held the model's OWN
notes with own content while they still under-counted 10-25 % of the time (a missed blink shows up as "off 1" where the label
says 2 or 3; 3 can never be an under-count). The loss at the onset is always the label's count, so the model learned "a note
saying 1 usually means 2" and "3 means 3": a correction for its own past errors, not a faithful read. By 1000 the loop had
closed further: episodes 25 and 29 now call the light-off phases "wait for the light" (the picture alone cannot tell "off after
k blinks" from "no blink yet"; only the bank can) and write that, so the last note before the go was "wait" and the go count
was the prior 2. The archived v4/500 checkpoint under the v4b rule is not a fallback: its own sentences at 0.4-0.5
confidence produce premature "off 1" notes at tick 2-3 and vocabulary garbage ("light on: 2 of 2 and carry", "<loc...>"
strings), so the 500-1000 stretch is what taught the model to write; it is the read side that was spoiled.

`pi05_yam_beans0922_v4c` (`V4C_TRUST_NOTES` = `V4B_WRITE_RULE` + `memory_v5_own_commit_label_content=True` +
`memory_v7_onset_ce_weight=6.0`), resumed from v4b's checkpoint 1000 (hard-linked into the v4c experiment directory, W&B run
pre-created) on the two H200s (v4b stopped at step ~1090 at 08:53, v4c restored 1000 and was compiling at 08:58; W&B run ybpotcga). Two existing generic mechanisms, no sentence or phase named: (1) the model still decides WHEN
to write (change, lowest-word gate 0.3, two-tick confirmation on its own sentences) but the bank receives the label sentence
of that tick (the v6.2 "B2" rule), so no under-counted note ever sits in the bank during training and the onset target can
never disagree with the note it should copy; a bank digit means what it says again. Deployment is unchanged: own timing, own
content, `v5_heldout_video.py` and the robot server write the model's own sentence. (2) The sentence loss on the ticks where
the label sentence changes carries weight 6 instead of 3 (the onset is one tick per window; the aggregate exact hides it).
New telemetry `onset_sentence_exact` (commit 459a44a) logs that tick alone. Trade-off accepted: a wrong own note at deployment
is now copied instead of "corrected", so the write side (blink misses, the "wait" confusion) is judged by the own-note videos
at every 250 steps; if under-counting persists the fix belongs to the writer (tick rate, confirmation length), not to the reader.

### 2026-09-23 14:40-16:31 -- why v4b/v4c fail at the go onset (onset A/B), and v4e

Tool: `openpi/scripts/v5_onset_ab.py` scores ONE training window at the go onset two ways -- the training loss path with the
count digit set to 1/2/3, and the rollout decode path (forced prefix, greedy) -- with switches for dropping note rows, blank
images, zeroed robot state, pointer off, window start and step. Launchers `beans/eval/h100_onset_ab*.sh`, `h200_onset_ab.sh`,
chains `h200_ab_seq*.sh`; logs `beans/eval/onset_ab_ep<ep>_<ckpt>_<condition>.log`. Both paths agree to 0.2 nats everywhere.

Findings at v4c/2000 (training episodes; x = scoop count):
- ep3 and ep8 (x=1): at the true go onset the model says "scoop 2 times" with p = 1.0 under every input change -- note rows
  dropped one by one or all, pointer off, images blanked, state zeroed, mid-go tick, window started 2/8/16 ticks earlier with
  the history teacher-forced inside the window. Only blank images AND no notes gives the null answer "3" (0.88). One single
  note "light off: 1 green blink" with blank images still gives 2. Digit 1 is never predicted at a true onset (8-15 nats).
- ep2 (x=2) -> 2 (0.78); ep0 (x=3) -> 3 (1.0). Images alone: 3 for ep0, 2 for ep3/ep8, soft 2 for ep2 -- an episode
  fingerprint (the tray's leftover beans mark the recording session), not a count cue (frames checked).
- v4/500 had p(1) = 0.48 at the same tick; v4b (own-content notes) turned it into 2 (0.999) and v4c did not undo it.
- The count battery over REAL training windows (`v5_count_flip_eval.py --split train`, 48 windows, 452 go steps): x=1
  correct 281/282 and still correct with the bank emptied. The true-count CE is ~0 on essentially every go step, i.e. these
  are copy ticks: a decision phase lasts ~38 ticks and every tick after the first restates the note already committed. The
  battery's "count_in_window" flag also never fires (it looks one token too early), so its "history only" summary meant
  nothing. Training grades the onset itself (weight 6) but it is a sliver of the phase's loss, while the light-phase onsets
  ("k blinks so far" -> "light on: k+1") outnumber the go onsets 2-3:1 and teach "onset digit = note digit + 1" -- which is
  exactly the observed map 1->2, 2->2/3 (0.22 on 3), 3->3.

v4e (`pi05_yam_beans0922_v4e`, commit ea22aa5), started 16:35 from the clean v4/500 (`archive/v4_500` hard-linked into the
v4e experiment directory, user's choice) on the two H200s (hgx-2 job 17425063 GPUs 0,1, batch 8), write rule = v4c. Two
existing generic knobs move the supervision onto the onsets: the transition-anchored start branch opens windows closer to
a sentence change and more often (`memory_critical_start_pad` 75 -> 25 frames, `memory_critical_prob` 0.5 -> 0.7 -- the
window begins with the notes prefilled and the change a few ticks ahead, the deployed situation), and the copy ticks of a
decision phase whose arm already moves are down-weighted (`memory_v6_decision_ce_weight_after_motion` 1.0 -> 0.2).
Success criterion, checked after every checkpoint by `beans/eval/eval_loop_v4e.sh` on the H100 (one line per checkpoint in
`beans/eval/v4e_onset_summary.log`): p(1) at the true onset of ep3/ep8 with prefill notes only must climb from ~0; if it is
still ~0 by 1000-1250 the "+1" habit dominates and the next change is on the read side, not more weight.
