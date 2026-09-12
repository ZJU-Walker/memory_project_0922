# boba_0911 — subtask label definition (v1, 2026-09-12 01:17 user decision; boundaries verified 02:10)

Data: `/iris/u/kewalk/memory_project/data/boba_0911/demo{1..60}` minus demo6 (moved to `boba_0911_excluded/`: the
video writer never finished, 2973/2709/2229 frames vs 4207 steps). 30 Hz, 4120–6083 frames (141–208 s), three
cameras 640x480 H.264, frame counts equal `metadata.json` num_steps on all 59. Bimanual: the LEFT arm handles the cup
(stack of cups top-left, tray in the middle, tea dispenser with a lever tap top-right); the RIGHT arm handles the
lids, the wooden spoons and the tap lever. Three identical lidded steel bins sit between the arms, fixed layout in
every episode: left = white sago, middle = black boba, right = red beans.

Episode protocol (identical in all 59): a human opens bin 1, 2, 3 in that order (stirring each with its spoon) and
closes them; the robot then makes drink 1 (cup on tray, lid off bin 2, 3 spoonfuls of boba into the cup, spoon back,
lid on, cup under the tap, right arm presses the tap ~10 s) and drink 2 (same with bin 3, 1 spoonful of red beans).
After each drink the arms reset and the human removes the cup. The cup sits on the tray under the tap while it is
filled; there is no "put the cup back" step.

## Sentences (21; one per frame; `subtask_labels.json` = list of {task, start, end}, inclusive, tiling [0, n-1])

| # | segment starts at | sentence |
|---|---|---|
| 1 | frame 0 | `watch, sago left bin` |
| 2 | bin 2 interior first visible in the top camera (human lifts lid 2) | `watch, boba middle bin` |
| 3 | bin 3 interior first visible | `watch, bean right bin` |
| 4 | first robot motion (round 1) / left-arm motion onset toward cup 2 (round 2) | `{first,second}, get cup` |
| 5 | left gripper releases the cup on the tray | `{first,second}, open {boba,bean} bin` |
| 6 | right gripper releases the lid (spoon grasp belongs here) | `first, scoop, 1 of 3` / `second, scoop, 1 of 1` |
| 7 | arm settles over the bin after pour k-1 (end of the swing back from the cup) | `first, scoop, k of 3` (k = 2, 3) |
| 8 | end of the last pour (wrist tilt back below the pour threshold) | `{first,second}, put the scoop back` |
| 9 | right gripper releases the spoon | `{first,second}, close {boba,bean} bin` |
| 10 | right gripper releases the lid after closing | `{first,second}, place cup` |
| 11 | left gripper releases the cup under the tap (or the left arm starts its return swing, whichever first) | `{first,second}, press tap to fill cup` |
| 12 | right gripper releases the tap lever; runs to the next `get cup` onset / last frame | `{first,second}, done` |

Rules follow cluster_v5/BEANS_LABELS.md v3: segments start at EVENTS; the retract after an event belongs to the next
segment (so `place cup` contains the right arm's retreat from the bin and the left arm's carry, `press tap` contains
the left arm's return and the right arm's approach to the lever, `done` contains the reset and the wait for the
human). The scoop count is stated in every scoop sentence (`k of 3`) — user decision for this task (the beans v3
"state x once" argument was noted; both tasks keep their own convention).

## Detection (scripts/boba_build_subtask_labels.py; all thresholds in the file header)

* Reveal onsets: bin-interior ROIs in the top camera (`ROIS`, 640x480 coords), classified W/K/R vs lid every 3rd
  frame; onset = first run of >= 3 content samples. Order 1 < 2 < 3 < first motion in all 59.
* Left arm: motion bursts (smoothed joint speed > 0.08 rad/s, gaps < 2 s merged). A = grasp a cup (full close,
  shoulder j1 max <= ~2.5), B = carry the cup under the tap (j1 max > 2.55, or > 2.4 with only a partial ~0.7 close).
  Cup-on-tray release = gripper back above 0.5; cup-under-tap release = gripper rising off its plateau in the 3 s
  before the return swing (else the swing onset).
* Right arm: gripper-closed runs (< 0.5, >= 8 frames). Gripper plateau separates the objects: lid handle 0.05–0.09,
  spoon 0.13–0.22, tap lever 0.01–0.02. Between the cup release and the place-cup burst the run order is
  `[failed lid grasp]* lid-open spoon-run+ lid-close`; lid-close is the last run, pours anchor the spoon runs.
* Pours: wrist joint rj4 > 0.15 for >= 8 frames, preceded by a dig (rj4 < -0.05) since the previous pour, counted
  only in spoon-plateau runs that are neither first nor last in the window. 3 in round 1 and 1 in round 2 in 56/59.
* Bin arrival for scoop k >= 2: after pour k-1 ends, the first frame the arm speed drops below 0.4 rad/s after the
  swing (> 0.8 rad/s).

Verification: per-episode sheets (top / left wrist / right wrist frame at every segment start) checked for
demo1/7/27/29/32/54/59 — every boundary lands on the intended event. Segment lengths over the 56 labelled episodes
(median s): watch 4.5/3.3/5.8, get cup 5.0, open bin 9.3/8.2, scoop 10.9/8.3/7.0 (round 2: 8.4), put back 1.6,
close bin 5.0/4.6, place cup 10.4, press tap 13.5/15.9, done 6.8/3.4.

## Excluded from labelling (protocol deviations; raw data kept in place, `include: false` in the manifest)

| demo | reason |
|---|---|
| demo20 | round 1 has only 2 spoonfuls of boba (verified on the top camera 29–48 s) |
| demo40 | drink 2 restarted: after the bean scoop the cup was not carried under the tap; a third cup was fetched and the whole second drink redone |
| demo49 | same as demo40 |

Manifest: `data/boba_0911/subtask_labels_manifest_boba.json` (per episode: status, notes, detected events, segments).
Re-run: `cd memory_project_v6/openpi && python3 scripts/boba_build_subtask_labels.py --write [--sheets DIR]`.
