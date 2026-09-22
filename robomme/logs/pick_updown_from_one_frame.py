import glob, json, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "cluster_robomme/eval"); import panda_fk
root = "../robomme/data/lerobot/PickXtimes_official"
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{root}/meta/tasks.jsonl")}
TICK, FLAT = 20, 0.01
rows = []
for f in sorted(glob.glob(f"{root}/data/chunk-000/*.parquet")):
    t = pq.read_table(f).to_pydict(); st = np.array(t["state"], np.float64); lab = [tasks[i] for i in t["task_index"]]
    ep = t["episode_index"][0]; pos = panda_fk.tcp_positions(st); z = pos[:, 2]; grip = st[:, 7]; zmin = z.min()
    # segment-relative time: frames since this pick label started
    start = 0
    for k in range(len(st) - TICK):
        if k > 0 and lab[k] != lab[k - 1]: start = k
        if not lab[k].startswith("pick up the"): continue
        rows.append((ep, k - start, pos[k, 0], pos[k, 1], z[k] - zmin, grip[k], z[k + TICK] - z[k]))
R = np.array(rows); ep, since, h, g, dz = R[:, 0], R[:, 1], R[:, 4], R[:, 5], R[:, 6]
move = np.abs(dz) >= FLAT; up = dz > 0
print("gripper width (cm) while going DOWN inside a pick: percentiles 5/25/50/75/95 =", np.round(np.percentile(g[move & ~up] * 100, [5, 25, 50, 75, 95]), 2))
print("gripper width (cm) while going UP   inside a pick: percentiles 5/25/50/75/95 =", np.round(np.percentile(g[move & up] * 100, [5, 25, 50, 75, 95]), 2))
X = np.c_[R[:, 2], R[:, 3], h, g * 10.0]  # continuous width, 1 cm of width = 10 cm of position
Xm, ym, em, hm, sm = X[move], up[move], ep[move], h[move], since[move]
pred = np.zeros_like(ym); conf = np.zeros(len(ym))
for e in np.unique(em):
    q = em == e; ref = ~q
    d = ((Xm[q][:, None, :] - Xm[ref][None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, 7, axis=1)[:, :7]; votes = ym[ref][nn].mean(1)
    pred[q] = votes > 0.5; conf[q] = np.abs(votes - 0.5) * 2
wrong = pred != ym
print(f"continuous width + tool xyz, leave-one-episode-out 7-NN: right {(1-wrong.mean())*100:.1f} % of {len(ym)} moving pick frames; wrong {wrong.sum()}")
at_cube = hm < 0.03
print(f"  wrong at the cube (tool < 3 cm above table): {(wrong & at_cube).sum()} of {at_cube.sum()} such frames ({(wrong & at_cube).mean()/max(at_cube.mean(),1e-9)*100:.1f} %)")
print(f"  wrong mid-air (>= 3 cm):                    {(wrong & ~at_cube).sum()} of {(~at_cube).sum()} such frames ({(wrong & ~at_cube).sum()/(~at_cube).sum()*100:.1f} %)")
print(f"  wrong in the first 20 frames of a pick label: {(wrong & (sm < 20)).sum()} | later: {(wrong & (sm >= 20)).sum()}")
# what the wrong mid-air ones are
w = wrong & ~at_cube
print(f"  wrong mid-air: going up {ym[w].sum()}, going down {(~ym[w]).sum()}; width median {np.median(Xm[w][:,3]/10*100):.2f} cm; height median {np.median(hm[w])*100:.1f} cm; frames-since-label median {np.median(sm[w]):.0f}")
# the one-frame ambiguity that matters for the policy: same (xyz, width) neighbourhood, both directions present -> how many such frames and what happens next?
amb = conf < 0.5
print(f"  neighbourhood split (< 5:2): {amb.sum()} frames ({amb.mean()*100:.1f} %), of them at the cube {(amb & at_cube).sum()}, mid-air {(amb & ~at_cube).sum()}")
