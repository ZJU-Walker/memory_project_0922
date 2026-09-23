"""Geometry of the token-after-token sentence bank (beans0922 v1/v3), on CPU, no data loader.

For each blink count x in 1..3 the true note sequence of an x-blink episode ("wait", "light on: 1", "light off: 1", ...,
"light off: x") is written into a fresh bank exactly as the policy writes (v6 token-level keys/values, one commit per note,
`--ticks-between` decay ticks between notes). Then:
  (A) digit values: cosines between the stored VALUE vectors of the tokens " 1", " 2", " 3" (is the count stored distinctly?);
  (B) ceiling read: the bank read with the exact context key of the digit position of "light off: <d> ..." -> cosine to the three
      digit values and the margin (what a perfect question would get); same with the go sentence's digit context
      ("yellow go: pick up the scoop, scoop <d> ...") BEFORE the go note exists (the onset case) and after it was written;
  (C) question read: the checkpoint's learned read questions (base questions, zero context shift) -> the 8 retrieved vectors per
      bank -> cosine between the answers for the 1-, 2- and 3-blink banks (how different do the memory tokens look for
      different counts?) and, per question, the nearest reference-token value.
  python beans/eval/token_bank_geometry.py --config-name pi05_yam_beans0922_v3 --params <ckpt>/params --output <json>
"""
import argparse, json, pathlib, re, sys
import numpy as np
import jax, jax.numpy as jnp
import sentencepiece

import openpi.models.model as _model
import openpi.shared.project_paths as project_paths
import openpi.training.config as _config
from openpi.models import memory as _memory

DIGITS = {1: 235274, 2: 235284, 3: 235304}  # " 1", " 2", " 3" (paligemma ids, as in v5_count_flip_eval)


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True); ap.add_argument("--params", type=pathlib.Path, required=True)
    ap.add_argument("--ticks-between", type=int, default=6, help="decay ticks between consecutive notes (light phase ~1 s = 6 ticks)")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--stage", choices=["led", "scoop"], default="led",
                    help="led = A/B/C above (the go onset); scoop = D: after the go note write the scoop-phase notes "
                         "('scoop k of x: dig and carry' / 'dump and return', 'done') with --scoop-ticks-between decay ticks "
                         "between them and, after each, read the go count with the exact go key, the newest scoop note's "
                         "'of x' key and the 8 questions (base, and base + last-note shift for v3) -> how different are the "
                         "answers for x=1/2/3 at the same scoop stage, and how much of the go note is left")
    ap.add_argument("--scoop-ticks-between", type=int, default=40, help="decay ticks between scoop-phase notes (~7 s at 6 ticks/s)")
    args = ap.parse_args()
    cfg = _config.get_config(args.config_name)
    params = _model.restore_params(args.params, restore_type=np.ndarray)
    model = cfg.model.load(params); model.eval()
    sp = sentencepiece.SentencePieceProcessor(model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model")))
    length = cfg.model.memory_v5_sentence_len
    rows = [tuple(int(t) for t in r) for r in cfg.model.memory_v5_reference_tokens]
    text = {r: sp.decode([t for t in r if t not in (108, 1)]).strip() for r in rows}
    by_text = {v: k for k, v in text.items()}
    def row_of(prefix):
        m = [t for t in by_text if t.startswith(prefix)]
        assert len(m) == 1, (prefix, m); return by_text[m[0]]
    def padded(row):
        tok = np.zeros((1, length), np.int32); msk = np.zeros((1, length), bool)
        tok[0, :len(row)] = row; msk[0, :len(row)] = True
        return jnp.asarray(tok), jnp.asarray(msk)
    def digit_pos(row):
        for i, t in enumerate(row):
            if t in DIGITS.values(): return i
        raise ValueError(row)

    write = lambda st, row, commit: model.v6_semantic_write_tokens(st, *padded(row), jnp.asarray([commit]))[0]
    read = lambda st, q: np.asarray(model.memory_semantic.read_key(st, jnp.asarray(q, jnp.float32)[None] if np.ndim(q) == 2 else jnp.asarray(q, jnp.float32)[None, None]))[0]
    dummy = row_of("wait for the light")

    # (A) digit values
    vals = {d: np.asarray(model.v6_token_values(*padded((DIGITS[d],))))[0, 0] for d in (1, 2, 3)}
    out = {"config": args.config_name, "params": str(args.params), "ticks_between": args.ticks_between,
           "digit_value_cos": {f"{a}-{b}": cos(vals[a], vals[b]) for a in (1, 2, 3) for b in (1, 2, 3) if a < b}}
    print("A) cos between digit VALUES:", out["digit_value_cos"], flush=True)

    # context keys of the digit position (identical for the three counts by construction -> take d=1 rows)
    off_row = {d: row_of(f"light off: {d} green") for d in (1, 2, 3)}
    go_row = {d: row_of(f"yellow go: pick up the scoop, scoop {d} time") for d in (1, 2, 3)}
    def ctx_key(row):
        keys, _, _ = model.v6_sentence_token_kv(*padded(row)); return np.asarray(keys)[0, digit_pos(row)]
    k_off = ctx_key(off_row[1]); k_go = ctx_key(go_row[1])
    out["context_key_cos"] = {"lightoff_d1_vs_d3": cos(k_off, ctx_key(off_row[3])), "lightoff_vs_go": cos(k_off, k_go)}
    print("   context keys: same 'light off:' context for d=1 vs d=3 ->", round(out["context_key_cos"]["lightoff_d1_vs_d3"], 3),
          "| 'light off:' vs 'scoop, scoop' ->", round(out["context_key_cos"]["lightoff_vs_go"], 3), flush=True)

    # base questions (zero context shift)
    base = np.asarray(model.memory_sem_read_query_bank.value, np.float32)
    q = np.asarray(_memory.l2_normalize(model.memory_sem_query_proj(jnp.asarray(base)).astype(jnp.float32)))  # [8, dk]
    ref_ids = list(model.v6_reference_token_ids())
    ref_vals = {t: np.asarray(model.v6_token_values(*padded((t,))))[0, 0] for t in ref_ids}
    def nearest(v):
        best = max(ref_ids, key=lambda t: cos(v, ref_vals[t])); return sp.decode([best]) or str(best), cos(v, ref_vals[best])

    if args.stage == "scoop":
        return scoop_stage(args, model, sp, out, row_of, padded, digit_pos, write, read, vals, k_off, k_go, q, nearest, ctx_key)

    banks, reads = {}, {}
    for x in (1, 2, 3):
        st = model.memory_semantic.init_state(1)
        seq = [row_of("wait for the light")]
        for k in range(1, x + 1):
            seq += [row_of(f"light on: {k} green"), row_of(f"light off: {k} green")]
        for i, row in enumerate(seq):
            st = write(st, row, True)
            for _ in range(args.ticks_between - 1): st = write(st, dummy, False)
        banks[x] = st
        r_off = read(st, k_off); r_go = read(st, k_go)
        st_after = write(st, go_row[x], True); r_go_after = read(st_after, k_go); r_off_after = read(st_after, k_off)
        res = {"lightoff_key_cos_to_digit": {d: cos(r_off, vals[d]) for d in (1, 2, 3)},
               "go_key_before_go_note_cos_to_digit": {d: cos(r_go, vals[d]) for d in (1, 2, 3)},
               "go_key_after_go_note_cos_to_digit": {d: cos(r_go_after, vals[d]) for d in (1, 2, 3)},
               "lightoff_key_after_go_note_cos_to_digit": {d: cos(r_off_after, vals[d]) for d in (1, 2, 3)},
               "lightoff_read_norm": float(np.linalg.norm(r_off)), "go_read_norm_before": float(np.linalg.norm(r_go))}
        R = read(st, q)  # [8, dv]
        reads[x] = R
        res["question_nearest_token"] = [nearest(R[i]) for i in range(R.shape[0])]
        res["question_read_norms"] = [float(np.linalg.norm(R[i])) for i in range(R.shape[0])]
        out[f"bank_x{x}"] = res
        print(f"B) x={x}: read with the 'light off:' key -> cos to digit 1/2/3 = "
              f"{[round(res['lightoff_key_cos_to_digit'][d], 3) for d in (1,2,3)]} | with the 'scoop, scoop' key BEFORE the go note: "
              f"{[round(res['go_key_before_go_note_cos_to_digit'][d], 3) for d in (1,2,3)]} (norm {res['go_read_norm_before']:.3f} vs {res['lightoff_read_norm']:.3f}) | AFTER: "
              f"{[round(res['go_key_after_go_note_cos_to_digit'][d], 3) for d in (1,2,3)]}", flush=True)
        print(f"C) x={x}: the 8 questions' nearest reference tokens: {[(t, round(c, 2)) for t, c in res['question_nearest_token']]}", flush=True)
    pairs = {}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        per_q = [cos(reads[a][i], reads[b][i]) for i in range(reads[a].shape[0])]
        pairs[f"x{a}_vs_x{b}"] = {"per_question": per_q, "all_8_tokens": cos(reads[a], reads[b])}
        print(f"C) cos between the 8-token answers for the x={a} and x={b} banks: per question {[round(c, 3) for c in per_q]} | pooled {pairs[f'x{a}_vs_x{b}']['all_8_tokens']:.3f}", flush=True)
    # the same comparison for the ceiling read (context key)
    ceil = {f"x{a}_vs_x{b}": cos(read(banks[a], k_off), read(banks[b], k_off)) for a, b in ((1, 2), (1, 3), (2, 3))}
    print("   for comparison, cos between the 'light off:' key reads of different banks:", {k: round(v, 3) for k, v in ceil.items()}, flush=True)
    out["question_read_cos_between_banks"] = pairs; out["lightoff_key_read_cos_between_banks"] = ceil
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", args.output)


def scoop_stage(args, model, sp, out, row_of, padded, digit_pos, write, read, vals, k_off, k_go, q, nearest, ctx_key):
    """D) the scoop phase: is the go count still readable after the scoop notes, and do the 8 answers differ by x?"""
    decay = lambda st, n: model.memory_semantic.analytic_decay(st, int(n))[0]
    shifted = getattr(model, "memory_v0920_query_context", False)
    base = jnp.asarray(np.asarray(model.memory_sem_read_query_bank.value, np.float32))
    def questions(prev_row):
        pre = model.memory_sem_query_proj(base).astype(jnp.float32)
        if shifted and prev_row is not None:
            tok, msk = padded(prev_row)
            emb = model.PaliGemma.llm(tok, method="embed").astype(jnp.float32)
            w = msk.astype(jnp.float32)[..., None]
            note = jnp.sum(emb * w, axis=1) / jnp.maximum(jnp.sum(w, axis=1), 1.0)
            pre = pre + model.memory_sem_query_prev_proj(note).astype(jnp.float32)
        return np.asarray(_memory.l2_normalize(pre))
    def scoop_seq(x):
        seq = []
        for k in range(1, x + 1):
            seq.append(("s%ddig" % k, row_of(f"scoop {k} of {x}: dig and carry")))
            if k < x:
                seq.append(("s%ddump" % k, row_of(f"scoop {k} of {x}: dump and return")))
        seq.append(("done", row_of("done, put down")))
        return seq
    stages = {}
    for x in (1, 2, 3):
        st = model.memory_semantic.init_state(1)
        led = [row_of("wait for the light")]
        for k in range(1, x + 1):
            led += [row_of(f"light on: {k} green"), row_of(f"light off: {k} green")]
        for row in led:
            st = write(st, row, True); st = decay(st, args.ticks_between - 1)
        go = row_of(f"yellow go: pick up the scoop, scoop {x} time")
        st = write(st, go, True)
        prev = go
        timeline = [("go", go)] + scoop_seq(x)
        for i, (name, row) in enumerate(timeline):
            if i > 0:
                st = decay(st, args.scoop_ticks_between - 1); st = write(st, row, True); prev = row
            r_go = read(st, k_go); r_off = read(st, k_off)
            rec = {"go_key_cos_to_digit": {d: cos(r_go, vals[d]) for d in (1, 2, 3)}, "go_key_read_norm": float(np.linalg.norm(r_go)),
                   "lightoff_key_cos_to_digit": {d: cos(r_off, vals[d]) for d in (1, 2, 3)}}
            if name != "go" and name != "done":
                # digit_pos finds the FIRST digit (k); the x digit is the second one
                pos = [j for j, t in enumerate(row) if t in DIGITS.values()]
                keys, _, _ = model.v6_sentence_token_kv(*padded(row)); k_of = np.asarray(keys)[0, pos[-1]]
                r_of = read(st, k_of)
                rec["newest_of_key_cos_to_digit"] = {d: cos(r_of, vals[d]) for d in (1, 2, 3)}
                rec["newest_of_read_norm"] = float(np.linalg.norm(r_of))
            R0 = read(st, q); rec["base_nearest"] = [nearest(R0[j]) for j in range(R0.shape[0])]
            rec["base_norms"] = [float(np.linalg.norm(R0[j])) for j in range(R0.shape[0])]
            ans = {"base": R0}
            if shifted:
                R1 = read(st, questions(prev)); rec["shifted_nearest"] = [nearest(R1[j]) for j in range(R1.shape[0])]; ans["shifted"] = R1
            stages.setdefault(name, {})[x] = (rec, ans)
            gk = rec["go_key_cos_to_digit"]; msg = f"D) x={x} stage {name}: exact go key -> cos to 1/2/3 {[round(gk[d],3) for d in (1,2,3)]} norm {rec['go_key_read_norm']:.3f}"
            if "newest_of_key_cos_to_digit" in rec:
                nk = rec["newest_of_key_cos_to_digit"]; msg += f" | newest 'of x' key -> {[round(nk[d],3) for d in (1,2,3)]}"
            msg += f" | base questions -> {[t for t, _ in rec['base_nearest']]}"
            if shifted: msg += f" | shifted -> {[t for t, _ in rec['shifted_nearest']]}"
            print(msg, flush=True)
    comp = {}
    for name, per_x in stages.items():
        xs = sorted(per_x)
        for a in xs:
            for b in xs:
                if a >= b: continue
                for kind in per_x[a][1]:
                    A, B = per_x[a][1][kind], per_x[b][1][kind]
                    per_q = [cos(A[j], B[j]) for j in range(A.shape[0])]
                    rel = float(np.linalg.norm(A - B) / (0.5 * (np.linalg.norm(A) + np.linalg.norm(B)) + 1e-12))
                    comp[f"{name}_x{a}_vs_x{b}_{kind}"] = {"per_question": per_q, "pooled": cos(A, B), "relative_difference": rel}
                    print(f"D) stage {name}: {kind} answers x={a} vs x={b}: pooled cos {cos(A, B):.3f} | relative difference {rel:.3f} | per question {[round(c, 3) for c in per_q]}", flush=True)
    out["scoop_stage"] = {name: {x: per_x[x][0] for x in per_x} for name, per_x in stages.items()}
    out["scoop_stage_answer_comparison"] = comp; out["scoop_ticks_between"] = args.scoop_ticks_between
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
