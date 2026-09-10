"""Which decision phrasing can READ the bin from the bank? (2026-09-09, user: replace closing + "open bin k" by one
sentence "lid closed, pick up <object> in bin k" held to the end.) The v6 pointer queries the bank with the key of the
context preceding the digit, so the digit is readable only if that context's key lands near the placement note's key.
For every episode: write the placement notes through the model's write path (with the decay gaps), then for each
candidate template read the digit of the prompted object and decode it among the digit tokens; report accuracy and the
cosine to the stored placement key.

  python scripts/v6_context_phrasing_probe.py --config-name pi05_yam_mem_v6_task1B3 --params <params> --manifest ... --sidecar ... --output out.txt
"""
import argparse, json, pathlib, re
import flax.nnx as nnx, jax, jax.numpy as jnp, numpy as np, sentencepiece
import openpi.models.model as _model
from openpi.shared import project_paths
from openpi.training import config as _config

TEMPLATES = {
    "closing note (today)": "{obj} in bin {k}",
    "open bin k (today's decision)": "open bin {k}",
    "user: lid closed, pick up OBJ in bin k": "lid closed, pick up {obj} in bin {k}",
    "lids closed, pick up the OBJ in bin k": "lids closed, pick up the {obj} in bin {k}",
    "note first: OBJ in bin k, lids closed, open it": "{obj} in bin {k}, lids closed, open it",
    "lids closed. OBJ in bin k": "lids closed. {obj} in bin {k}",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True); ap.add_argument("--params", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True); ap.add_argument("--sidecar", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True); ap.add_argument("--split", default="all")
    a = ap.parse_args()
    cfg = _config.get_config(a.config_name)
    model = cfg.model.load(_model.restore_params(a.params, restore_type=np.ndarray)); model.eval()
    stride = int(cfg.data.base_config.memory_stride_frames); sent_len = cfg.model.memory_v5_sentence_len
    sp = sentencepiece.SentencePieceProcessor(model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model")))
    ids_of = lambda s: sp.encode(s.lower().strip() + "\n")
    ids_raw = lambda s: sp.encode(s.lower().strip())

    def row(ids):
        tok = np.zeros((1, sent_len), np.int32); tok[0, :len(ids)] = ids
        m = np.zeros((1, sent_len), bool); m[0, :len(ids)] = True
        return jnp.asarray(tok), jnp.asarray(m)

    @nnx.jit
    def _write(model, s, t, m): return model.v6_semantic_write_tokens(s, t, m, jnp.ones((1,), bool))[0]
    @nnx.jit
    def _decay(model, s, g): return model.memory_semantic.analytic_decay(s, g)[0]
    @nnx.jit
    def _keys(model, t, m): return model.v6_sentence_token_kv(t, m)[0]
    @nnx.jit
    def _read(model, s, q): return model.memory_semantic.read_key(s, q)

    digit_ids = {k: ids_of(f"x in bin {k}")[-2] for k in (1, 2, 3)}  # token before the "\n"
    assert len(set(digit_ids.values())) == 3, digit_ids
    dvals = np.asarray(model.v6_token_values(jnp.asarray([list(digit_ids.values())], jnp.int32), jnp.ones((1, 3), bool)))[0]
    dkeys = list(digit_ids.keys())

    def digit_index(sentence, prefix_text):
        # "... bin 2" tokenizes as [..., "▁bin", "▁", "2"]: the digit sits one past the prefix, after the bare "▁" token
        ids = ids_of(sentence); pre = ids_raw(prefix_text)
        if ids[:len(pre)] != pre:
            return ids, None
        idx = len(pre)
        while idx < len(ids) and sp.id_to_piece(ids[idx]) == "\u2581":
            idx += 1
        return ids, idx

    manifest = json.loads(a.manifest.read_text()); sidecar = json.loads(a.sidecar.read_text())
    eps = [e for e in manifest["episodes"] if e.get("include", True) and (a.split == "all" or e.get("split") == a.split)]
    stats = {t: {"n": 0, "ok": 0, "cos_same": [], "cos_other": []} for t in TEMPLATES}
    skipped = 0
    for e in eps:
        segs = sidecar["episodes"][e["stable_id"]]["segments"]
        closing = segs[-2]["sentence"]; m = re.match(r"^(\w+) in bin (\d)$", closing)
        if not m:
            skipped += 1; continue
        obj, k = m.group(1), int(m.group(2))
        notes = segs[:-2]
        state = model.memory_semantic.init_state(1); stored = {}
        for i, seg in enumerate(notes):
            ids = ids_of(seg["sentence"]); tok, mask = row(ids)
            state = _write(model, state, tok, mask)
            mm = re.match(r"^(\w+) in bin (\d)$", seg["sentence"])
            if mm:
                stored[mm.group(1)] = np.asarray(_keys(model, tok, mask))[0, len(ids) - 2]
            nxt = notes[i + 1]["start"] if i + 1 < len(notes) else seg["end"] + 1
            gap = max(int(round((nxt - seg["start"]) / stride)) - 1, 0)
            if gap:
                state = _decay(model, state, jnp.asarray([gap], jnp.int32))
        for name, tpl in TEMPLATES.items():
            sent = tpl.format(obj=obj, k=k); prefix = sent[: sent.index(f"bin {k}") + 3]
            ids, idx = digit_index(sent, prefix)
            if idx is None or ids[idx] != digit_ids[k]:
                continue
            tok, mask = row(ids)
            keys = np.asarray(_keys(model, tok, mask))[0]
            q = keys[idx]
            r = np.asarray(_read(model, state, jnp.asarray(q[None, None])))[0, 0]
            pred = dkeys[int(np.argmax(dvals @ r))]
            st = stats[name]; st["n"] += 1; st["ok"] += int(pred == k)
            if obj in stored:
                st["cos_same"].append(float(q @ stored[obj] / (np.linalg.norm(q) * np.linalg.norm(stored[obj]) + 1e-8)))
                for o2, k2 in stored.items():
                    if o2 != obj:
                        st["cos_other"].append(float(q @ k2 / (np.linalg.norm(q) * np.linalg.norm(k2) + 1e-8)))
    lines = [f"config {a.config_name} params {a.params}", f"episodes {len(eps)} (skipped {skipped})", ""]
    for name, st in stats.items():
        cs = np.mean(st["cos_same"]) if st["cos_same"] else float("nan"); co = np.mean(st["cos_other"]) if st["cos_other"] else float("nan")
        lines.append(f"{name:48s} digit read {st['ok']}/{st['n']} = {st['ok']/max(st['n'],1):.3f} | key cos to own note {cs:.3f}, to other objects' notes {co:.3f}")
    a.output.write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
