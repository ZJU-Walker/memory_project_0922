"""Recall through the MODEL's v6 write path (cluster_v6/README.md §4): for every episode of a task1 manifest, write
the label notes of the human phase (watching + the placements, in order, with the rollout's decay between them)
with `v6_semantic_write_tokens`, then read the digit of `<object> in bin _` for every placed object with the key of
that context (`v6_sentence_token_kv`, the same key function the write used) and score the read against the three
digit tokens' values. This is the parameter-free same-frame read the bank-level probe measured at 1.00; here it
runs with the trained checkpoint's projections. Also reports the pointer scale beta and the pointer-query norm."""

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

DIGITS = {1: 235274, 2: 235284, 3: 235304}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_yam_mem_v6_task1A")
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--sidecar", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="all", help="all | development | train")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--whiten-keys", action="store_true",
                        help="linear bank only: whiten the keys with the covariance of the reference token-context keys "
                             "(all positions of the config reference sentences), then unit-normalize")
    parser.add_argument("--no-decay", action="store_true", help="linear bank only: no per-step decay between notes")
    parser.add_argument("--layer-note", default="", help="free text recorded in the report header")
    parser.add_argument("--bank", choices=("model", "linear"), default="model",
                        help="model: the checkpoint Titans MLP bank through v6_semantic_write_tokens; linear: a plain "
                             "delta-rule matrix (the bank-level probe bank) on the SAME model keys/values and decay")
    args = parser.parse_args()

    cfg = _config.get_config(args.config_name)
    model = cfg.model.load(_model.restore_params(args.params, restore_type=np.ndarray))
    model.eval()
    stride = cfg.data.create(cfg.assets_dirs, cfg.model).memory_stride_frames
    sent_len = cfg.model.memory_v5_sentence_len
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )

    def row(sentence: str):
        ids = sp.encode(sentence.lower().strip() + "\n")
        tok = np.zeros((1, sent_len), dtype=np.int32); tok[0, : len(ids)] = ids
        mask = np.zeros((1, sent_len), dtype=bool); mask[0, : len(ids)] = True
        return jnp.asarray(tok), jnp.asarray(mask), ids

    beta = float(np.asarray(model.memory_v6_pointer_beta.value))
    qnorm = float(np.linalg.norm(np.asarray(model.memory_v6_pointer_query_proj.kernel.value)))
    knorm = float(np.linalg.norm(np.asarray(model.memory_v6_token_key_proj.kernel.value)))
    digit_tokens = jnp.asarray([[DIGITS[1], DIGITS[2], DIGITS[3]]], dtype=jnp.int32)
    digit_values = np.asarray(model.v6_token_values(digit_tokens, jnp.ones((1, 3), dtype=bool)))[0]  # [3, dv]

    @nnx.jit
    def _write(model, s, t, m):
        return model.v6_semantic_write_tokens(s, t, m, jnp.ones((1,), dtype=bool))[0]

    @nnx.jit
    def _decay(model, s, g):
        return model.memory_semantic.analytic_decay(s, g)[0]

    @nnx.jit
    def _keys_of(model, t, m):
        return model.v6_sentence_token_kv(t, m)[0]

    @nnx.jit
    def _read(model, s, q):
        return model.memory_semantic.read_key(s, q)

    write = lambda s, t, m: _write(model, s, t, m)  # noqa: E731
    decay = lambda s, g: _decay(model, s, g)  # noqa: E731
    keys_of = lambda t, m: _keys_of(model, t, m)  # noqa: E731
    read = lambda s, q: _read(model, s, q)  # noqa: E731

    @nnx.jit
    def _kv(model, t, m):
        k, v, _ = model.v6_sentence_token_kv(t, m)
        return k, v

    @nnx.jit
    def _hidden(model, s, k):
        return model.memory_semantic.hidden_key(s, k)

    alpha = float(model.memory_semantic.config.alpha_step)
    d_key = int(model.memory_semantic.config.d_key)
    manifest = json.loads(args.manifest.read_text()); sidecar = json.loads(args.sidecar.read_text())
    cos_key, cos_hidden = [], []
    # whitening map over the reference token-context keys (linear bank variant)
    whiten = None
    if args.whiten_keys:
        ref_tok, ref_mask = model.v5_reference_token_rows(sent_len)
        ref_keys = np.asarray(_keys_of(model, ref_tok, ref_mask))
        ref_ctx = ref_keys[np.asarray(ref_mask)]  # [n_contexts, dk]
        mu = ref_ctx.mean(0)
        cov = np.cov(ref_ctx - mu, rowvar=False) + 1e-3 * np.eye(ref_ctx.shape[1], dtype=np.float32)
        evals, evecs = np.linalg.eigh(cov)
        wmap = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-6))) @ evecs.T
        whiten = lambda k: (lambda z: z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9))((k - mu) @ wmap)  # noqa: E731
        print(f"whitening over {len(ref_ctx)} reference contexts; eigenvalue range {evals.min():.2e}..{evals.max():.2e}")
    episodes = [e for e in manifest["episodes"] if e.get("include", True) and (args.split == "all" or e["split"] == args.split)]
    lines, n_ok, n_all, tgt_ok, tgt_all, by_back = [], 0, 0, 0, 0, collections.defaultdict(lambda: [0, 0])
    margins = []
    for e in episodes:
        segs = sidecar["episodes"][e["stable_id"]]["segments"]
        notes = segs[:-2]  # watching + placements
        closing = segs[-2]
        state = model.memory_semantic.init_state(1)
        W = np.zeros((d_key, digit_values.shape[1]), dtype=np.float32)  # linear bank (probe's Bank)
        for i, seg in enumerate(notes):
            tok, mask, ids = row(seg["sentence"])
            nxt = notes[i + 1]["start"] if i + 1 < len(notes) else closing["start"]
            gap = max(int(round((nxt - seg["start"]) / stride)) - 1, 0)
            if args.bank == "model":
                state = write(state, tok, mask)
                if gap:
                    state = decay(state, jnp.asarray([gap], dtype=jnp.int32))
            else:
                k_all, v_all = _kv(model, tok, mask)
                k_all, v_all = np.asarray(k_all)[0, : len(ids)], np.asarray(v_all)[0, : len(ids)]
                if whiten is not None:
                    k_all = whiten(k_all)
                for kk, vv in zip(k_all, v_all, strict=True):  # sequential delta writes, rate 1, then the decay
                    W += np.outer(kk, vv - W.T @ kk)
                if not args.no_decay:
                    W *= (1.0 - alpha) ** (gap + 1)
        # key / hidden-feature geometry of the digit contexts of this episode
        dk = []
        for sent in [s_["sentence"] for s_ in notes[1:]]:
            tok, mask, ids = row(sent)
            dk.append(np.asarray(keys_of(tok, mask))[0, len(ids) - 2])
        dk = np.stack(dk)
        hk = np.asarray(_hidden(model, model.memory_semantic.init_state(1), jnp.asarray(dk[None])))[0]
        hk = hk / np.maximum(np.linalg.norm(hk, axis=-1, keepdims=True), 1e-9)
        iu = np.triu_indices(len(dk), 1)
        cos_key.append(float(np.mean((dk @ dk.T)[iu]))); cos_hidden.append(float(np.mean((hk @ hk.T)[iu])))
        placements = [s["sentence"] for s in notes[1:]]
        target = e["prompt"].split()[-1]
        results = []
        for j, sent in enumerate(placements):
            obj, k = sent.split(" in bin ")
            tok, mask, ids = row(sent)
            d = len(ids) - 2  # the digit sits before the trailing newline
            q = keys_of(tok, mask)[:, d : d + 1]
            if args.bank == "model":
                r = np.asarray(read(state, q))[0, 0]
            else:
                qn = np.asarray(q)[0, 0]
                if whiten is not None:
                    qn = whiten(qn[None])[0]
                r = W.T @ qn
            scores = digit_values @ r
            pred = int(np.argmax(scores)) + 1
            top = np.sort(scores)[::-1]
            ok = pred == int(k)
            back = len(placements) - 1 - j
            n_ok += ok; n_all += 1; by_back[back][0] += ok; by_back[back][1] += 1; margins.append(float(top[0] - top[1]))
            if obj == target:
                tgt_ok += ok; tgt_all += 1
            results.append(f"{obj}:{k}->{pred}{'' if ok else '!'}(m={top[0]-top[1]:.3f})")
        lines.append(f"{e['stable_id'].split('/')[-1]:14s} {e['split']:11s} target={target:6s} " + " ".join(results))
    summary = [
        f"checkpoint {args.params}  bank={args.bank} whiten_keys={args.whiten_keys} no_decay={args.no_decay} {args.layer_note}",
        f"digit-context keys of one episode: mean pairwise cosine in key space {np.mean(cos_key):.3f}, in the bank's hidden-feature space {np.mean(cos_hidden):.3f}",
        f"pointer beta={beta:.4f} |W_q|={qnorm:.3f} |P_k|={knorm:.3f}",
        f"model-path recall, all placed objects: {n_ok}/{n_all} = {n_ok / max(n_all, 1):.3f}; median margin {np.median(margins):.3f}",
        f"target objects only: {tgt_ok}/{tgt_all} = {tgt_ok / max(tgt_all, 1):.3f}",
        "by distance from the newest note: " + ", ".join(f"{b} back {v[0]}/{v[1]}" for b, v in sorted(by_back.items())),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(summary + ["", *lines]) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
