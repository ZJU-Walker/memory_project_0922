import h5py, sys
p = sys.argv[1]
with h5py.File(p, "r") as f:
    eps = sorted(f.keys())
    print("groups:", eps[:3], "...", len(eps))
    g = f[eps[0]]
    steps = sorted((k for k in g if k.startswith("timestep_")), key=lambda s: int(s.split("_")[1]))
    print("timesteps:", len(steps), "other:", [k for k in g if not k.startswith("timestep_")])
    if "setup" in g:
        for k in g["setup"]:
            v = g["setup"][k][()]
            print(f"  setup/{k}: {getattr(v, 'shape', '')} {str(v)[:160]}")
    s = g[steps[0]]
    def walk(h, pre=""):
        for k in h:
            if isinstance(h[k], h5py.Group):
                walk(h[k], pre + k + "/")
            else:
                d = h[k]
                print(f"  {pre}{k} {d.shape} {d.dtype}")
    walk(s)
    labels = []
    for k in steps:
        v = g[k]["info"]["simple_subgoal"][()]
        labels.append(v.decode() if isinstance(v, bytes) else str(v))
    segs = [(labels[0], 0)]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            segs.append((labels[i], i))
    print("segments:", [(s, i) for s, i in segs])
