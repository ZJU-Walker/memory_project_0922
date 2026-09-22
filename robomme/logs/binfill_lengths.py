import h5py, sys, collections, statistics as st
p = sys.argv[1]
with h5py.File(p, "r") as f:
    eps = sorted(k for k in f.keys() if k.startswith("episode_"))
    lens = {}
    for e in eps:
        lens[e] = sum(1 for k in f[e].keys() if k.startswith("timestep_"))
    L = sorted(lens.values())
    print("episodes", len(L), "min/med/max", L[0], st.median(L), L[-1])
    for thr in (600, 700, 800, 900, 1000, 1050):
        print(f"  frames > {thr}: {sum(l > thr for l in L)}")
    print("  sorted lengths:", L)
    runs = []
    for e in eps[::10]:
        g = f[e]
        keys = sorted((k for k in g.keys() if k.startswith("timestep_")), key=lambda s: int(s.split("_")[1]))
        prev, n = None, 0
        for k in keys:
            s = g[k]["info"]["simple_subgoal"][()]
            s = s.decode() if isinstance(s, bytes) else str(s)
            if s == prev:
                n += 1
            else:
                if prev is not None:
                    runs.append((prev, n))
                prev, n = s, 1
        runs.append((prev, n))
    ln = [n for _, n in runs]
    print("segments in 10 episodes:", len(ln), "min/med/max", min(ln), st.median(ln), max(ln))
    for thr in (15, 20, 25, 30, 40):
        print(f"  segments < {thr} frames: {sum(n < thr for n in ln)}")
    by = collections.defaultdict(list)
    for s, n in runs:
        by[s].append(n)
    for s in sorted(by):
        v = by[s]
        print(f"  {s!r}: n={len(v)} med={st.median(v)} min={min(v)}")
