"""Spectral / Frobenius norms of v3's question maps: base P_q, image-context Z_ctx, last-note Z_prev (CPU, params only)."""
import sys, pathlib, numpy as np, openpi.models.model as _model
params = _model.restore_params(pathlib.Path(sys.argv[1]), restore_type=np.ndarray)
def find(tree, name, path=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from find(v, name, path + (k,))
    elif name in path:
        yield path, tree
for name in ("memory_sem_query_proj", "memory_sem_query_context_proj", "memory_sem_query_prev_proj", "memory_sem_read_query_bank"):
    for path, arr in find(params, name):
        a = np.asarray(arr, np.float32)
        s = np.linalg.svd(a, compute_uv=False) if a.ndim == 2 else None
        print("/".join(path), a.shape, f"fro {np.linalg.norm(a):.3f}", f"spectral {s[0]:.3f} rank-1 share {(s[0]**2 / (s**2).sum()):.2f}" if s is not None else "", flush=True)
