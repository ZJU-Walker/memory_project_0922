"""List every string/Path field of pi05_robomme_0920_v0 that is an existing local path resolving OUTSIDE this tree."""
import dataclasses, os, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from openpi.training import config as _config
root = pathlib.Path("/iris/u/kewalk/memory_project_0920").resolve()
cfg = _config.get_config("pi05_robomme_0920_v0")
seen = []
def walk(value, name):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            walk(getattr(value, f.name), f"{name}.{f.name}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): walk(v, f"{name}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items(): walk(v, f"{name}[{k!r}]")
    elif isinstance(value, (str, pathlib.Path)):
        s = str(value)
        if s.startswith("/") and os.path.exists(s):
            r = pathlib.Path(s).resolve()
            inside = str(r).startswith(str(root) + "/")
            seen.append((name, s, str(r), inside))
walk(cfg, "cfg")
for name, s, r, inside in seen:
    flag = "OK " if inside else "OUT"
    print(f"{flag} {name} = {s}" + ("" if r == s else f"  -> {r}"))
print("out-of-tree:", sum(not x[3] for x in seen))
