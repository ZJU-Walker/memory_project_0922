#!/usr/bin/env python3
"""Build per-episode `subtask_labels.json` for boba_0911 from the arm signals + top-camera bin colours.

Task (59 episodes, 30 Hz, ~4600 frames): a human reveals the three lidded bins (left: white sago, middle: black
boba, right: red beans) and closes them; the robot then makes two drinks. Left arm: get a cup onto the tray, later
carry it under the tea tap. Right arm: lift the lid, scoop with the wooden spoon into the cup (3x boba for the
first drink, 1x red bean for the second), put the spoon back, close the lid, press the tap lever.

Vocabulary (22 sentences, user decision 2026-09-12 01:17):
  watch, sago left bin | watch, boba middle bin | watch, bean right bin
  {first|second}, get cup
  {first|second}, open {boba|bean} bin
  first, scoop, {1,2,3} of 3 | second, scoop, 1 of 1
  {first|second}, put the scoop back
  {first|second}, close {boba|bean} bin
  {first|second}, place cup
  {first|second}, press tap to fill cup
  {first|second}, done

Boundary rules (segments start at EVENTS, the retract after an event belongs to the next segment, as in
cluster_v5/BEANS_LABELS.md v3):
  watch, X          reveal onsets from the bin-interior colour in the top camera (bin 1 label runs from frame 0)
  get cup           first robot motion (round 1) / left-arm motion onset toward the second cup (round 2)
  open bin          left gripper releases the cup on the tray
  scoop 1           right gripper releases the lid (spoon grasp is part of scoop 1)
  scoop k>=2        arm settles over the bin after pour k-1 (end of the swing back from the cup)
  put the scoop back  end of the last pour (wrist tilt rj4 drops back below the pour threshold)
  close bin         right gripper releases the spoon
  place cup         right gripper releases the lid after closing
  press tap         left gripper releases the cup under the tap (or the left arm starts its return, whichever first)
  done              right gripper releases the tap lever; runs until the next round's get-cup onset / episode end

Output: <demo>/subtask_labels.json (list of {task,start,end}, inclusive frames, tiling [0,n-1]) and
<data>/subtask_labels_manifest_boba.json (per episode: num_frames, events, segments, status, notes).
"""
import argparse, json, os, pathlib, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

DATA = pathlib.Path("/iris/u/kewalk/memory_project/data/boba_0911")
FPS = 30
ROIS = {1: (245, 300, 278, 345), 2: (296, 300, 328, 345), 3: (351, 300, 388, 345)}  # bin interiors, top cam 640x480
ROI_STEP = 3
ROUNDS = [("first", "boba", 3), ("second", "bean", 1)]
WATCH = ["watch, sago left bin", "watch, boba middle bin", "watch, bean right bin"]

# thresholds (all verified on the joint traces of demo1/2/30/51/52/54/55/56/59)
L_MOVE = 0.08        # rad/s smoothed joint speed -> left arm "moving"
GRIP_CLOSED = 0.5    # gripper < this = holding (cup on tray, lid, spoon, tap)
LID_PLATEAU = 0.11   # right gripper plateau: lid handle < 0.11 <= spoon handle < 0.45 ; tap lever ~0.01
SPOON_PLATEAU = 0.45
POUR_RJ4 = 0.15      # right wrist joint 4 above this = spoon tilted over the cup
DIG_RJ4 = -0.05      # a pour must be preceded by the wrist below this (digging) since the previous pour
SWING_FAST, SWING_SLOW = 0.8, 0.4   # rad/s: swing back from the cup / settled over the bin
B_REACH = 2.55       # left shoulder joint 1 max during a burst: > this = the arm reached the tap (place cup)


def runs(mask, minlen=1):
    m = np.asarray(mask).astype(bool)
    d = np.diff(np.r_[0, m.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0] - 1
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a + 1 >= minlen]


def speed(j, win=15):
    v = np.linalg.norm(np.diff(j[:, :6], axis=0), axis=1) * FPS
    v = np.r_[v, v[-1]]
    return np.convolve(v, np.ones(win) / win, mode="same")


def classify_rgb(m):
    r, g, b = m; br = m.mean(); chroma = m.max() - m.min()
    if br > 150 and chroma < 70: return "W"
    if br < 45: return "K"
    if r - max(g, b) > 18 and br < 130: return "R"
    if 55 < br < 105 and chroma < 25: return "L"
    return "?"


def reveal_onsets(ep: pathlib.Path, upto: int):
    """Per bin: first frame with >=3 consecutive content samples (W/K/R) in the top camera, before `upto`."""
    nfr = upto // ROI_STEP + 1
    cmd = ["ffmpeg", "-v", "error", "-i", str(ep / "top_camera_rgb.mp4"), "-vf",
           f"select='not(mod(n\\,{ROI_STEP}))'", "-vsync", "vfr", "-frames:v", str(nfr),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 480, 640, 3)
    onsets, classes = {}, {}
    for k, (x0, y0, x1, y1) in ROIS.items():
        lab = [classify_rgb(fr[i, y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(0)) for i in range(len(fr))]
        content = np.array([c in "WKR" for c in lab])
        rr = runs(content, 3)
        onsets[k] = int(rr[0][0] * ROI_STEP) if rr else None
        cs = [c for c in lab if c in "WKR"]
        classes[k] = max(set(cs), key=cs.count) if cs else "?"
    return onsets, classes


def left_bursts(lj, lg):
    sp = speed(lj)
    bs = []
    for a, b in runs(sp > L_MOVE):
        if bs and a - bs[-1][1] < 2 * FPS: bs[-1][1] = b
        else: bs.append([a, b])
    out = []
    for a, b in bs:
        if b - a < FPS: continue
        g = lg[a:b + 1]; gmin = float(g.min()); reach = float(lj[a:b + 1, 1].max())
        # A = grasp a cup from the stack (full close, reach <= ~2.5); B = carry the cup under the tap (reach > 2.55,
        # gripper only partially closed on the wide cup, ~0.65-0.75, sometimes with a brief full close first)
        kind = "B" if (reach > B_REACH or (reach > 2.4 and gmin > 0.45)) else ("A" if gmin < 0.3 else "X")
        ev = {"onset": a, "end": b, "gmin": round(gmin, 3), "reach": round(reach, 2), "kind": kind}
        if kind == "A":
            grasp = a + int(np.argmax(g < GRIP_CLOSED))
            rel = grasp + int(np.argmax(lg[grasp:] > GRIP_CLOSED))
            ev.update(grasp=grasp, release=rel)
        if kind == "B":
            # cup release under the tap = the gripper rising off its plateau within 3 s before the return swing
            # (the last fast motion of the burst); if no rise is visible, the return-swing onset itself.
            fast = runs(sp[a:b + 1] > SWING_FAST)
            ret = a + fast[-1][0] if fast else b
            w0 = max(a, ret - 3 * FPS); w = lg[w0:ret + 1]
            rel = None
            if len(w) > 2 and lg[ret] - w.min() >= 0.08:
                i_min = int(np.argmin(w)); rel = w0 + i_min + int(np.argmax(w[i_min:] > w.min() + 0.5 * (lg[ret] - w.min())))
            ev.update(return_onset=ret, grip_release=rel, release=rel if rel is not None else ret)
        out.append(ev)
    return out


def right_runs(rg):
    return [{"a": a, "b": b, "plateau": float(np.median(rg[a:b + 1]))} for a, b in runs(rg < GRIP_CLOSED, 8)]


def pours(rj, a, b):
    """Pour excursions (rj4 > POUR_RJ4) inside [a,b], each preceded by a dig (rj4 < DIG_RJ4)."""
    w = rj[a:b + 1, 4]
    ex = runs(w > POUR_RJ4, 8)
    merged = []
    for s, e in ex:
        if merged and s - merged[-1][1] < 20: merged[-1][1] = e
        else: merged.append([s, e])
    out, last = [], 0
    for s, e in merged:
        if (w[last:s] < DIG_RJ4).any():
            out.append((a + s, a + e)); last = e
    return out


def bin_arrival(rj, after, limit):
    sp = speed(rj)
    i = after
    while i < limit and sp[i] < SWING_FAST: i += 1
    while i < limit and sp[i] > SWING_SLOW: i += 1
    return i if i < limit else None


def build(ep: pathlib.Path):
    lj = np.load(ep / "left_joint_positions.npy"); rj = np.load(ep / "right_joint_positions.npy")
    lg = np.load(ep / "left_gripper_position.npy")[:, 0]; rg = np.load(ep / "right_gripper_position.npy")[:, 0]
    n = min(len(lj), len(rj), len(lg), len(rg))
    notes, ev = [], {"num_frames": n}
    lb = left_bursts(lj, lg)
    A = [x for x in lb if x["kind"] == "A"]; B = [x for x in lb if x["kind"] == "B"]
    ev["left_bursts"] = lb
    if len(A) != 2 or len(B) != 2:
        notes.append(f"left bursts: {len(A)} grasp(A) / {len(B)} place(B) bursts, expected 2/2")
    rsp = speed(rj); r_first = int(np.argmax(rsp > 0.05))
    fm = min(A[0]["onset"] if A else n, r_first)
    ev["first_motion"] = fm
    ons, cls = reveal_onsets(ep, fm + 15)
    ev["reveal_onsets"] = ons; ev["reveal_classes"] = cls
    if not (ons[1] is not None and ons[2] is not None and ons[3] is not None and ons[1] < ons[2] < ons[3] < fm):
        notes.append(f"reveal onsets not ordered before first motion: {ons} fm={fm}")
    rr = right_runs(rg); ev["right_runs"] = rr
    segs = []
    if len(A) == 2 and len(B) == 2 and ons[2] and ons[3] and ons[1] is not None and ons[1] < ons[2] < ons[3] < fm:
        segs += [(WATCH[0], 0), (WATCH[1], ons[2]), (WATCH[2], ons[3])]
        rounds = []
        for k, (word, bin_name, nsc) in enumerate(ROUNDS):
            a_, b_ = A[k], B[k]
            start = fm if k == 0 else a_["onset"]
            nxt = A[k + 1]["onset"] if k + 1 < len(A) else n
            win = [r for r in rr if r["a"] >= a_["release"] and r["b"] < b_["onset"]]
            rd = {"round": word, "get_cup_start": start, "cup_release": a_["release"]}
            # structure of the window between the cup release and the place-cup burst:
            #   [failed lid grasp]* lid-open  spoon-run+ (regrasps allowed)  lid-close
            # so the lid-close is always the LAST run, and pours (only possible while holding the spoon) anchor the
            # spoon window; lid runs are excluded as pour candidates by position and gripper plateau.
            if len(win) < 3:
                notes.append(f"{word}: only {len(win)} right-gripper runs in window"); rounds.append(rd); continue
            for r in win: r["pours"] = []
            for r in win[1:-1]:
                if r["plateau"] >= LID_PLATEAU: r["pours"] = pours(rj, r["a"], r["b"])
            pour_runs = [r for r in win if r["pours"]]
            if not pour_runs:
                notes.append(f"{word}: no pour found in any right-gripper run"); rounds.append(rd); continue
            p0, p1 = pour_runs[0], pour_runs[-1]
            lid_open = [r for r in win if r["b"] < p0["a"] and r["plateau"] < LID_PLATEAU]
            if not lid_open:
                notes.append(f"{word}: no lid-open run with a lid plateau before the first pour")
                rounds.append(rd); continue
            lo = lid_open[-1]; lc = win[-1]
            if lc is p1:
                notes.append(f"{word}: last run before place-cup contains a pour (no lid close?)"); rounds.append(rd); continue
            if lc["b"] - lc["a"] > 9 * FPS or lc["plateau"] >= 0.2:
                notes.append(f"{word}: run before place-cup does not look like a lid close: {(lc['a'], lc['b'], round(lc['plateau'],2))}")
            sp_runs = [r for r in win if lo["b"] < r["a"] < lc["a"]]
            spoon_a, spoon_b = sp_runs[0]["a"], sp_runs[-1]["b"]
            odd = [(r["a"], r["b"], round(r["plateau"], 2)) for r in sp_runs if not (LID_PLATEAU <= r["plateau"] < SPOON_PLATEAU)]
            if odd: notes.append(f"{word}: spoon-window runs with a non-spoon gripper plateau: {odd}")
            pr = [p for r in sp_runs for p in r["pours"]]
            rd.update(lid_open=(lo["a"], lo["b"]), spoon=(spoon_a, spoon_b), lid_close=(lc["a"], lc["b"]), pours=pr,
                      spoon_runs=[(r["a"], r["b"]) for r in sp_runs])
            if len(pr) != nsc:
                notes.append(f"{word}: {len(pr)} pours detected, expected {nsc}: {pr}")
            taps = [r for r in rr if r["a"] >= b_["release"] and r["a"] < nxt and r["b"] - r["a"] >= 2 * FPS]
            if not taps: notes.append(f"{word}: no tap press run after place-cup"); rounds.append(rd); continue
            tp = taps[0]
            rd.update(tap=(tp["a"], tp["b"]), place_release=b_["release"], place_return=b_["return_onset"],
                      place_grip_release=b_["grip_release"])
            rounds.append(rd)
            if len(pr) != nsc: continue
            segs.append((f"{word}, get cup", start))
            segs.append((f"{word}, open {bin_name} bin", a_["release"]))
            segs.append((f"{word}, scoop, 1 of {nsc}", lo["b"] + 1))
            arr = []
            for i in range(1, nsc):
                s = bin_arrival(rj, pr[i - 1][1] + 1, pr[i][0])
                if s is None:
                    notes.append(f"{word}: no bin arrival found before pour {i+1}; using pour end + 1 s"); s = pr[i - 1][1] + FPS
                arr.append(s); segs.append((f"{word}, scoop, {i+1} of {nsc}", s))
            rd["arrivals"] = arr
            segs.append((f"{word}, put the scoop back", pr[-1][1] + 1))
            segs.append((f"{word}, close {bin_name} bin", spoon_b + 1))
            segs.append((f"{word}, place cup", lc["b"] + 1))
            segs.append((f"{word}, press tap to fill cup", b_["release"]))
            segs.append((f"{word}, done", tp["b"] + 1))
        ev["rounds"] = rounds
    # tile
    out, ok = [], not notes
    starts = [s for _, s in segs]
    if ok and (len(segs) != 3 + 10 + 8 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= n):
        notes.append(f"segment starts not strictly increasing / wrong count ({len(segs)}): {starts}"); ok = False
    if ok:
        for i, (task, s) in enumerate(segs):
            e = segs[i + 1][1] - 1 if i + 1 < len(segs) else n - 1
            out.append({"task": task, "start": int(s), "end": int(e)})
    return {"demo": ep.name, "num_frames": n, "status": "ok" if ok else "REVIEW", "notes": notes, "events": ev, "segments": out}


def to_jsonable(o):
    if isinstance(o, dict): return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o


def sheet(ep: pathlib.Path, res: dict, out_dir: pathlib.Path):
    """One JPG per episode: for every segment start, the top / left-wrist / right-wrist frame at that frame."""
    from PIL import Image, ImageDraw
    segs = res["segments"]
    if not segs:  # REVIEW episode: show the raw events instead
        ev = res["events"]; frames = []
        for r in ev.get("rounds", []):
            for key in ("get_cup_start", "cup_release"): frames.append((f"{r['round']} {key}", r[key]))
            for key in ("lid_open", "spoon", "lid_close", "tap"):
                if key in r: frames.append((f"{r['round']} {key}", r[key][0]))
            for i, p in enumerate(r.get("pours", [])): frames.append((f"{r['round']} pour{i+1}", p[0]))
        frames = [(t, f) for t, f in frames if f is not None]
    else:
        frames = [(s["task"], s["start"]) for s in segs]
    if not frames: return
    idx = sorted(set(f for _, f in frames))
    sel = "+".join(f"eq(n\\,{f})" for f in idx)
    tiles = {}
    for cam in ("top", "left", "right"):
        cmd = ["ffmpeg", "-v", "error", "-i", str(ep / f"{cam}_camera_rgb.mp4"), "-vf", f"select='{sel}',scale=240:180",
               "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        arr = np.frombuffer(raw, np.uint8).reshape(-1, 180, 240, 3)
        tiles[cam] = {f: arr[i] for i, f in enumerate(idx[:len(arr)])}
    cols = 2; rows = (len(frames) + cols - 1) // cols
    W, H = 720, 196
    im = Image.new("RGB", (cols * W, rows * H), "black"); d = ImageDraw.Draw(im)
    for i, (task, f) in enumerate(frames):
        x0, y0 = (i % cols) * W, (i // cols) * H
        for j, cam in enumerate(("top", "left", "right")):
            if f in tiles[cam]: im.paste(Image.fromarray(tiles[cam][f]), (x0 + j * 240, y0 + 16))
        d.text((x0 + 4, y0 + 2), f"{ep.name}  f={f} ({f/FPS:.1f}s)  {task}", fill="yellow")
    im.save(out_dir / f"{ep.name}.jpg", quality=85)


def process(args):
    ep, write, sheets_dir = args
    res = build(ep)
    if write and res["status"] == "ok":
        (ep / "subtask_labels.json").write_text(json.dumps(res["segments"], indent=2) + "\n")
    if sheets_dir:
        try: sheet(ep, res, sheets_dir)
        except Exception as e: res["notes"].append(f"sheet failed: {e}")
    return to_jsonable(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--write", action="store_true", help="write <demo>/subtask_labels.json for episodes with status ok")
    ap.add_argument("--sheets", default=None, help="directory for per-episode verification sheets")
    ap.add_argument("--manifest", default=None, help="manifest path (default <data>/subtask_labels_manifest_boba.json)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    data = pathlib.Path(a.data)
    demos = a.demos or sorted([p.name for p in data.glob("demo*") if p.is_dir()], key=lambda s: int(s[4:]))
    sheets_dir = pathlib.Path(a.sheets) if a.sheets else None
    if sheets_dir: sheets_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(process, [(data / d, a.write, sheets_dir) for d in demos]))
    vocab = sorted({s["task"] for r in results for s in r["segments"]})
    man = {"data": str(data), "vocabulary": vocab, "episodes": {r["demo"]: r for r in results}}
    mp = pathlib.Path(a.manifest) if a.manifest else data / "subtask_labels_manifest_boba.json"
    mp.write_text(json.dumps(man, indent=1) + "\n")
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"{n_ok}/{len(results)} episodes ok; vocabulary {len(vocab)}; manifest -> {mp}")
    for r in results:
        if r["status"] != "ok":
            print(f"  REVIEW {r['demo']}: " + " | ".join(r["notes"]))
        else:
            d = [f"{s['task'].split(', ',1)[1] if ', ' in s['task'] else s['task']}:{(s['end']-s['start']+1)/FPS:.0f}" for s in r["segments"]]
            print(f"  ok {r['demo']:7s} n={r['num_frames']} " + " ".join(d))


if __name__ == "__main__":
    main()
