"""Task-agnostic recall probe through the MODEL's v6 write path (cluster_v6/README.md §7).

For every episode of a v5 generic manifest + sidecar: write the label sentences in order, with the rollout's decay
between them, up to (not including) the first DECISION sentence (`memory_required_subtasks` of the data config);
then, for every token context that was written (the tokens before position p >= 1 of a written sentence), ask the
bank for the token most recently written under that context, by decoding the read against the values of every
token of the reference vocabulary. Reports accuracy overall, on the VARIABLE contexts (a prefix that the reference
sentences continue with more than one token: the digits, the object words, ...) and by age (how many distinct
sentences were written after the last write of that context; 0 = it is in the newest sentence). A bank that keeps
older notes apart scores ~1.0 at every age; a newest-wins bank scores at chance for age >= 1 on the variable slots.
No task knowledge is used: the sentence list, the decision sentences and the tokenizer all come from the config."""

import argparse
import collections
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

from openpi.models import model as _model
from openpi.shared import project_paths
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--sidecar", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--decisions", default="",
                        help="comma-separated decision sentences (default: the data config's memory_required_subtasks)")
    args = parser.parse_args()

    cfg = _config.get_config(args.config_name)
    model = cfg.model.load(_model.restore_params(args.params, restore_type=np.ndarray))
    model.eval()
    dc = cfg.data.base_config
    stride = int(dc.memory_stride_frames)
    decisions = {d.strip() for d in args.decisions.split(",") if d.strip()} or set(dc.memory_required_subtasks or ())
    if not decisions:
        raise SystemExit("no decision sentences: pass --decisions")
    sent_len = cfg.model.memory_v5_sentence_len
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )

    def ids_of(sentence: str) -> list[int]:
        return sp.encode(sentence.lower().strip() + "\n")

    def row(ids):
        tok = np.zeros((1, sent_len), dtype=np.int32); tok[0, : len(ids)] = ids
        mask = np.zeros((1, sent_len), dtype=bool); mask[0, : len(ids)] = True
        return jnp.asarray(tok), jnp.asarray(mask)

    @nnx.jit
    def _write(model, s, t, m):
        return model.v6_semantic_write_tokens(s, t, m, jnp.ones((1,), dtype=bool))[0]

    @nnx.jit
    def _decay(model, s, g):
        return model.memory_semantic.analytic_decay(s, g)[0]

    @nnx.jit
    def _keys(model, t, m):
        return model.v6_sentence_token_kv(t, m)[0]

    @nnx.jit
    def _read(model, s, q):
        return model.memory_semantic.read_key(s, q)

    # reference vocabulary: every token of the reference sentences; "variable" prefixes = continued by > 1 token
    ref_rows = [tuple(r) for r in cfg.model.memory_v5_reference_tokens]
    vocab = sorted({t for r in ref_rows for t in r})
    vocab_values = np.asarray(model.v6_token_values(jnp.asarray([vocab], dtype=jnp.int32), jnp.ones((1, len(vocab)), dtype=bool)))[0]
    continuations = collections.defaultdict(set)
    for r in ref_rows:
        for p in range(1, len(r)):
            continuations[r[:p]].add(r[p])
    variable = {prefix for prefix, nxt in continuations.items() if len(nxt) > 1}

    manifest = json.loads(args.manifest.read_text()); sidecar = json.loads(args.sidecar.read_text())
    episodes = [e for e in manifest["episodes"] if e.get("include", True) and (args.split == "all" or e.get("split") == args.split)]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    tot = collections.Counter(); ok = collections.Counter(); lines = []
    for e in episodes:
        segs = sidecar["episodes"][e["stable_id"]]["segments"]
        cut = next((i for i, g in enumerate(segs) if g["sentence"] in decisions), len(segs))
        notes = segs[:cut]
        if not notes:
            continue
        state = model.memory_semantic.init_state(1)
        latest: dict[tuple, tuple[int, int]] = {}  # prefix -> (token, index of the sentence that wrote it last)
        for i, seg in enumerate(notes):
            ids = ids_of(seg["sentence"])
            tok, mask = row(ids)
            state = _write(model, state, tok, mask)
            for p in range(1, len(ids)):
                latest[tuple(ids[:p])] = (ids[p], i)
            nxt = notes[i + 1]["start"] if i + 1 < len(notes) else seg["end"] + 1
            gap = max(int(round((nxt - seg["start"]) / stride)) - 1, 0)
            if gap:
                state = _decay(model, state, jnp.asarray([gap], dtype=jnp.int32))
        n_notes = len(notes)
        wrong = []
        for prefix, (expected, idx) in latest.items():
            tok, mask = row(list(prefix) + [expected])
            q = _keys(model, tok, mask)[:, len(prefix) : len(prefix) + 1]
            r = np.asarray(_read(model, state, q))[0, 0]
            pred = vocab[int(np.argmax(vocab_values @ r))]
            age = n_notes - 1 - idx
            var = prefix in variable
            for key in ("all", f"age{min(age, 3)}") + (("variable", f"variable_age{min(age, 3)}") if var else ()):
                tot[key] += 1; ok[key] += int(pred == expected)
            if pred != expected and var:
                wrong.append(f"{sp.decode(list(prefix))!r}->{sp.decode([expected])!r} read {sp.decode([pred])!r} (age {age})")
        lines.append(f"{e['stable_id'].split('/')[-1]:16s} notes={n_notes} variable-slot misses: {wrong if wrong else 'none'}")

    def frac(k):
        return f"{ok[k]}/{tot[k]} = {ok[k] / tot[k]:.3f}" if tot[k] else "n/a"
    summary = [
        f"config {args.config_name}  params {args.params}",
        f"bank dims {model.memory_semantic.config.dims}  whiten_keys={getattr(model, 'memory_v6_whiten_keys', False)}  "
        f"episodes {len(lines)}  reference vocabulary {len(vocab)} tokens  variable prefixes {len(variable)}",
        f"ALL contexts: {frac('all')}   by age: " + ", ".join(f"{a}: {frac('age' + str(a))}" for a in range(4)),
        f"VARIABLE contexts: {frac('variable')}   by age: " + ", ".join(f"{a}: {frac('variable_age' + str(a))}" for a in range(4)),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(summary + ["", *lines]) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
