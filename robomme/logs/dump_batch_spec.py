"""One real loader batch of <config> (exactly as probe_0920_data.py builds it: config batch/workers, CPU JAX) saved as a pickle
of numpy arrays (sample 0 only) so the GPU node can build abstract batch shapes for an ahead-of-time compile.
usage: dump_batch_spec.py <config> <out.pkl>"""
import os, pickle, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np
from openpi.training import config as _config, data_loader as _dl
name, out = sys.argv[1], sys.argv[2]
t0 = time.time(); cfg = _config.get_config(name)
loader = _dl.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=1, exact_resume=False)
print(f"loader built in {time.time()-t0:.0f}s", flush=True)
obs, actions = next(iter(loader))
print(f"batch in {time.time()-t0:.0f}s: batch {cfg.batch_size}", flush=True)
one = jax.tree.map(lambda x: np.asarray(x)[:1], (obs, actions))
for path, leaf in jax.tree_util.tree_flatten_with_path(one)[0]:
    print(jax.tree_util.keystr(path), leaf.shape, leaf.dtype)
with open(out, "wb") as f:
    pickle.dump(one, f)
print("saved", out, flush=True)
