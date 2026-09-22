"""Ahead-of-time compile of the real train step (same jit / shardings / donation as scripts/train.py) at a given batch size
and print XLA's memory analysis per device: temp (activations + workspace), arguments (params + optimizer + batch), output,
aliased (donated) bytes. No training, no weights loaded, ~no GPU memory (XLA_PYTHON_CLIENT_PREALLOCATE=false).
usage: aot_memory.py <config> <batch> <batch_spec.pkl> [fsdp=4]"""
import dataclasses, functools, os, pickle, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_compilation_cache", False)  # the NFS cache corrupts under concurrent compiles
sys.path.insert(0, "scripts")
import train as T  # noqa: E402
from openpi.training import config as _config, sharding  # noqa: E402

name, batch, pkl = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fsdp = int(sys.argv[4]) if len(sys.argv) > 4 else 4
cfg = dataclasses.replace(_config.get_config(name), batch_size=batch, fsdp_devices=fsdp)
T._configure_v35_runtime_environment(cfg)  # noqa: SLF001 (same XLA / precision environment as a real run)
mesh = sharding.make_mesh(fsdp)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
state_shape, state_sharding = T.init_train_state(cfg, jax.random.key(0), mesh, resume=True)
with open(pkl, "rb") as f:
    one = pickle.load(f)
batch_spec = jax.tree.map(lambda x: jax.ShapeDtypeStruct((batch,) + tuple(x.shape[1:]), x.dtype, sharding=data_sharding), one)
state_spec = jax.tree.map(lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh), state_shape, state_sharding)
tokens = {k: v.shape for k, v in batch_spec[0].images.items()}
print(f"{name} batch {batch} fsdp {fsdp}: images {tokens} actions {batch_spec[1].shape}", flush=True)
step = jax.jit(functools.partial(T.train_step, cfg), in_shardings=(replicated, state_sharding, data_sharding),
               out_shardings=(state_sharding, replicated), donate_argnums=(1,))
t0 = time.time()
with sharding.set_mesh(mesh):
    compiled = step.lower(jax.random.key(0), state_spec, batch_spec).compile()
ma = compiled.memory_analysis()
gb = lambda b: b / 2**30  # noqa: E731
print(f"RESULT {name} batch {batch}: temp {gb(ma.temp_size_in_bytes):.1f} GiB | args {gb(ma.argument_size_in_bytes):.1f} GiB | "
      f"output {gb(ma.output_size_in_bytes):.1f} GiB | aliased {gb(ma.alias_size_in_bytes):.1f} GiB | "
      f"peak ~ temp + args + (output - aliased) = {gb(ma.temp_size_in_bytes + ma.argument_size_in_bytes + ma.output_size_in_bytes - ma.alias_size_in_bytes):.1f} GiB "
      f"| compile {time.time() - t0:.0f}s", flush=True)
