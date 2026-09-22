"""Model parameter tree of pi05_robomme_0920_v0 (eval_shape, no allocation) vs the original pi05_base checkpoint metadata,
classified with the config's loader allowlists. Everything the base lacks must be a memory leaf (fresh init)."""
import os, re
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, flax.nnx as nnx
from flax import traverse_util
import orbax.checkpoint as ocp
from openpi.training import config as _config
from openpi.training import robomme_config as rc

cfg = _config.get_config("pi05_robomme_0920_v0")
state = jax.eval_shape(lambda rng: nnx.state(cfg.model.create(rng)), jax.random.key(0))
pure = nnx.to_pure_dict(state) if hasattr(nnx, "to_pure_dict") else state.to_pure_dict()
model = {"/".join(str(k) for k in path): v for path, v in traverse_util.flatten_dict(pure).items() if v is not None}
meta = ocp.PyTreeCheckpointer().metadata(cfg.weight_loader.params_path if hasattr(cfg.weight_loader, "params_path") else cfg.weight_loader.checkpoint_path)
tree = meta.tree if hasattr(meta, "tree") else meta
src = {"/".join(str(k) for k in path).removeprefix("params/"): v for path, v in traverse_util.flatten_dict(tree).items()}
print("model leaves", len(model), "| source leaves", len(src))
non_mem, mem = re.compile(rc.NON_MEMORY_LEAF), re.compile(rc.MEMORY_LEAF)
only_model = sorted(k for k in model if k not in src)
only_src = sorted(k for k in src if k not in model)
mism = [(k, tuple(model[k].shape), tuple(getattr(src[k], "shape", ()))) for k in model if k in src and tuple(model[k].shape) != tuple(getattr(src[k], "shape", ()))]
print(f"in model only (must be fresh = memory leaves): {len(only_model)}")
bad_fresh = [k for k in only_model if not mem.fullmatch(k)]
print("  NOT matching MEMORY_LEAF:", bad_fresh)
print("  fresh sample:", only_model[:12])
print(f"in source only (loader has ignored_source_allowlist=() -> must be empty): {len(only_src)}", only_src[:20])
print(f"shape mismatches: {len(mism)}", mism[:10])
matched = [k for k in model if k in src]
print("matched but NOT allowed by NON_MEMORY_LEAF:", [k for k in matched if not non_mem.fullmatch(k)][:10])
import math
n_params = sum(math.prod(v.shape) for v in model.values())
print(f"total params {n_params/1e9:.3f} B")
