import h5py, collections, sys
f = h5py.File('/iris/u/kewalk/robomme_benchmark/data/robomme_data_h5/record_dataset_BinFill.h5', 'r')
names = sorted(f, key=lambda s: int(s.split("_")[-1]))
ep = f[names[0]]
print("setup keys:", list(ep['setup'].keys()))
for k in ep['setup'].keys():
    v = ep['setup'][k]
    try: print("  ", k, v[()] if v.shape == () or v.size < 8 else v.shape)
    except Exception as e: print("  ", k, "?", e)
t = ep['timestep_0']
def walk(g, prefix=""):
    for k, v in g.items():
        if isinstance(v, h5py.Group): walk(v, prefix + k + "/")
        else: print("  ", prefix + k, v.shape, v.dtype, (v[()] if v.shape == () else ""))
walk(t)
lens = []; vocab = collections.Counter(); goals = collections.Counter(); segs_per_ep = []
for name in names:
    g = f[name]
    n = len([k for k in g.keys() if k.startswith("timestep_")]); lens.append(n)
    goals[str(g['setup/task_goal'][()])] += 1
    if int(name.split("_")[-1]) % 10 == 0:
        labels = [g[f"timestep_{i}/info/simple_subgoal"][()] for i in range(0, n, 5)]
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
        for l in labels: vocab[l] += 1
        segs_per_ep.append(1 + sum(1 for a, b in zip(labels, labels[1:]) if a != b))
print("episodes", len(lens), "frames", sum(lens), "min/median/max", min(lens), sorted(lens)[50], max(lens))
print("goals (distinct)", len(goals)); [print("  ", g, c) for g, c in list(goals.items())[:8]]
print("segments per sampled episode (5-frame subsample):", segs_per_ep)
print("simple_subgoal vocabulary from every 10th episode:", len(vocab)); [print("  ", repr(s), c) for s, c in sorted(vocab.items())]
