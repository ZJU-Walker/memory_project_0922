"""Relative change of the fresh (memory) leaves between two checkpoints of the same run, read directly from the orbax/ocdbt
store (no model build). usage: memory_param_drift.py <exp dir> <step_a> <step_b>"""
import json, pathlib, sys
import numpy as np, tensorstore as ts
exp = pathlib.Path(sys.argv[1]); a, b = sys.argv[2], sys.argv[3]
man = json.load(open(exp / "initialization_graft_manifest.json"))["leaves"]
leaves = man["fresh_initialized"] + sorted([l for l in man["matched"] if l.get("shape")], key=lambda l: -__import__("math").prod(l["shape"]))[:6]  # + the 6 largest backbone leaves for scale
fresh_paths = {l["path"] for l in man["fresh_initialized"]}
def read(step, path):
    base = str(exp / step / "params")
    for key in (f"params.{path.replace('/', '.')}.value", f"params.{path.replace('/', '.')}", path.replace("/", "."), path):
        try:
            spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": base}, "path": key},
                    "recheck_cached_data": False, "recheck_cached_metadata": False}
            return np.asarray(ts.open(spec, open=True).result().read().result(), dtype=np.float64), key
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"{path}: {err}")
rows = []; total_a2 = total_d2 = 0.0
for leaf in leaves:
    p = leaf["path"]
    try:
        xa, key = read(a, p); xb, _ = read(b, p)
    except RuntimeError as e:
        continue  # biases absent from the store (orbax drops None slots)
    na = float(np.linalg.norm(xa)); nd = float(np.linalg.norm(xb - xa)); total_a2 += na * na; total_d2 += nd * nd
    rows.append((nd / max(na, 1e-12), nd, na, xa.size, ("fresh " if p in fresh_paths else "BACKBONE ") + p))
    if p not in fresh_paths: total_a2 -= na * na; total_d2 -= nd * nd
print(f"{sum(1 for r in rows if r[4].startswith('fresh'))} fresh leaves (+6 backbone) compared, steps {a} -> {b}; key form: {key}")
print(f"overall relative change of the memory leaves: {np.sqrt(total_d2) / np.sqrt(total_a2):.4%}  (|delta| {np.sqrt(total_d2):.4f} / |w| {np.sqrt(total_a2):.2f})")
print(f"{'rel change':>10} {'|delta|':>9} {'|w|':>9} {'n':>10}  leaf")
for r in sorted(rows, key=lambda r: -r[0])[:16]: print(f"{r[0]:10.4%} {r[1]:9.4f} {r[2]:9.3f} {r[3]:10,}  {r[4]}")
print("smallest:"); [print(f"{r[0]:10.4%} {r[1]:9.4f} {r[2]:9.3f} {r[3]:10,}  {r[4]}") for r in sorted(rows, key=lambda r: r[0])[:5]]
