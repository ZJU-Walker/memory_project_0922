"""Onset A/B (2026-09-23): same window, same bank, two ways of asking the model for the go count.

Path A (the count battery's way): Pi0._compute_sequence_loss_v32, teacher forced, the go sentence of the window's first
step rewritten with count 1 / 2 / 3 (v5_count_flip_eval.set_count), ranked by the step's sentence CE.  A second family of
variants masks every sentence token AFTER the digit, so the ranking then reflects the digit token alone.
Path B (the video rollout's way): the decode machinery of v5_heldout_video (memory-extended cache, step mask, pointer
bonus), fed the same variant tokens one by one, reporting the log-probability of every token and the distribution over the
three digits at the digit position; plus the free greedy decode.  The bank for path B is built exactly as the training
scan initialises a window: the label-history prefill rows at full strength with their gaps, the pending sentence as prev.

  python scripts/v5_onset_ab.py --config-name pi05_yam_beans0922_v4c --params .../1750/params --episode-index 3
"""
# ruff: noqa: I001
import pyarrow.parquet  # noqa: F401  isort: skip

import argparse
import json
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

import openpi.models.model as _model
import openpi.shared.project_paths as project_paths
import openpi.training.config as _config
import openpi.training.data_loader as data_loader_lib
from openpi.shared import nnx_utils

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import v5_count_flip_eval as battery  # noqa: E402
import v5_heldout_video as video  # noqa: E402

DIGITS = (battery.DIGIT_TOKENS[1], battery.DIGIT_TOKENS[2], battery.DIGIT_TOKENS[3])


def make_score_fn(model):
    """Path B: the rollout decode machinery with FORCED tokens; returns per-position log p(forced token) and the
    log-probabilities of the three digit tokens at every position."""

    @nnx.jit
    def score(model, observation, sem_state, prev_tokens, prev_mask, forced_tokens, forced_mask):
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        num_img = prefix_len - model.max_token_len
        top_tokens = model._top_camera_token_count(num_img, preprocessed.images)  # noqa: SLF001
        mem_len = model._memory_token_total  # noqa: SLF001
        gen_base = prefix_len + mem_len
        prepared = model._v0920_prepare_prefix(  # noqa: SLF001
            prefix_tokens, prefix_mask, prefix_ar, sem_state, top_token_count=top_tokens,
            visual_state=model.memory.init_state(batch), state=preprocessed.state,
            prev_tokens=prev_tokens, prev_mask=prev_mask,
        )
        kv_cache = prepared["cache"]
        final_prefix = prepared["final_prefix"]
        memory_valid = prepared["memory_valid"]
        pointer_on = bool(getattr(model, "memory_v6_pointer_read", False))
        context_pointer = pointer_on and getattr(model, "memory_v6_pointer_query", "hidden") == "context"
        s_len = model.memory_v5_sentence_len
        length = forced_tokens.shape[1]
        digit_ids = jnp.asarray(DIGITS, dtype=jnp.int32)

        def logits_of(hidden_vec, index, so_far_tokens, so_far_mask):
            logits = model.PaliGemma.llm(hidden_vec[:, None], method="decode")[:, 0].astype(jnp.float32)
            if pointer_on:
                on_span = jnp.broadcast_to(jnp.asarray(index) < s_len, (batch, 1))
                queries = None
                if context_pointer:
                    ctx_q = model.v6_context_queries(so_far_tokens[:, :s_len], so_far_mask[:, :s_len])
                    idx = jnp.clip(jnp.asarray(index), 0, s_len - 1)
                    queries = jax.lax.dynamic_index_in_dim(ctx_q, idx, axis=1, keepdims=True)
                logits = logits + model.v6_pointer_bonus(
                    hidden_vec[:, None], sem_state, on_span, logits.shape[-1], queries=queries
                )[:, 0]
            return logits

        def record(logits, i, logp, dd):
            lsm = jax.nn.log_softmax(logits, axis=-1)
            tok = jax.lax.dynamic_index_in_dim(forced_tokens, i, axis=1, keepdims=False)
            logp = logp.at[:, i].set(jnp.take_along_axis(lsm, tok[:, None], axis=-1)[:, 0])
            dd = dd.at[:, i].set(lsm[:, digit_ids])
            return logp, dd

        logp = jnp.zeros((batch, length), dtype=jnp.float32)
        dd = jnp.zeros((batch, length, 3), dtype=jnp.float32)
        empty_mask = jnp.zeros_like(forced_mask)
        logits0 = logits_of(
            model._v32_causal_seed(final_prefix, prefix_mask)[:, 0], 0, jnp.zeros_like(forced_tokens), empty_mask  # noqa: SLF001
        )
        logp, dd = record(logits0, 0, logp, dd)

        def step(carry, i):
            cache, logp, dd = carry
            previous = jax.lax.dynamic_index_in_dim(forced_tokens, i - 1, axis=1, keepdims=False)
            token_emb = model.PaliGemma.llm(previous[:, None], method="embed")
            step_attn = model._v32_step_mask(prefix_mask, i, memory_valid=memory_valid)  # noqa: SLF001
            (out, _), cache = model.PaliGemma.llm(
                [token_emb, None], mask=step_attn,
                positions=jnp.broadcast_to(gen_base + i - 1, (batch, 1)), kv_cache=cache, cache_position=gen_base + i - 1,
            )
            so_far_mask = forced_mask & (jnp.arange(length)[None, :] < i)
            so_far_tokens = jnp.where(so_far_mask, forced_tokens, 0)
            logits = logits_of(out[:, 0], i, so_far_tokens, so_far_mask)
            logp, dd = record(logits, i, logp, dd)
            return (cache, logp, dd), None

        (kv_cache, logp, dd), _ = jax.lax.scan(step, (kv_cache, logp, dd), jnp.arange(1, length))
        return logp, dd

    return score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", required=True)
    p.add_argument("--params", type=pathlib.Path, required=True)
    p.add_argument("--episode-index", type=int, required=True)
    p.add_argument("--frame", type=int, default=None, help="window start frame (default: the episode's first go frame)")
    p.add_argument("--manifest", type=pathlib.Path, default=None)
    p.add_argument("--sidecar", type=pathlib.Path, default=None)
    p.add_argument("--step", type=int, default=0, help="evaluate this step of the window instead of its first step (the bank stays the window prefill: under the two-tick confirmation nothing is committed before step 2)")
    p.add_argument("--drop-note-prefix", action="append", default=[], help="drop prefill rows whose text starts with this prefix (both paths), e.g. 'light on'")
    p.add_argument("--blank-images", action="store_true", help="replace every camera image with zeros (both paths)")
    p.add_argument("--zero-state", action="store_true", help="zero the robot state (proprio) for the whole window (both paths)")
    p.add_argument("--pointer-beta", type=float, default=None, help="override memory_v6_pointer_beta for BOTH paths (0 = no pointer bonus)")
    args = p.parse_args()

    cfg = _config.get_config(args.config_name)
    params = _model.restore_params(args.params, restore_type=np.ndarray)
    model = cfg.model.load(params)
    model.eval()
    if hasattr(model, "memory_v6_pointer_beta"):
        if args.pointer_beta is not None:
            model.memory_v6_pointer_beta.value = jnp.asarray(float(args.pointer_beta), dtype=jnp.float32)
        print("pointer beta:", float(model.memory_v6_pointer_beta.value), flush=True)
    print("decision_ce_weight_after_motion:", getattr(cfg.model, "memory_v6_decision_ce_weight_after_motion", None), "onset weight:", getattr(cfg.model, "memory_v7_onset_ce_weight", None), flush=True)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader_lib.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = data_loader_lib.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, *data_config.model_transforms.inputs]
    )
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )
    manifest_path = args.manifest or project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    sidecar = json.loads((args.sidecar or project_paths.project_path(project_paths.V5_SUBTASK_LABELS)).read_text())
    episodes = sorted([e for e in manifest["episodes"] if e.get("include", True)], key=lambda e: e["episode_index"])
    episode = episodes[args.episode_index]
    start = sum(int(e["expected_num_frames"]) for e in episodes[: args.episode_index])
    segments = sidecar["episodes"][episode["stable_id"]]["segments"]
    go_frame = next(s["start"] for s in segments if s["sentence"].startswith("yellow go"))
    frame = args.frame if args.frame is not None else go_frame
    stride = data_config.memory_stride_frames
    print(f"episode {args.episode_index} {episode['stable_id']} x={episode.get('class')} first go frame {go_frame}, window start {frame}", flush=True)

    # the window (retry earlier starts on the E-anchor refusal, like the rollout)
    first_t = 0
    while True:
        try:
            item = tds[start + frame - first_t * stride]
            break
        except ValueError as err:
            if "E anchor" not in str(err):
                raise
            first_t += 1
    if first_t:
        print(f"window fetched from frame {frame - first_t * stride} (the go step is step {first_t})", flush=True)
    batched = jax.tree.map(lambda x: np.asarray(x)[None], item)
    obs = _model.Observation.from_dict(batched)
    actions = jnp.asarray(batched["actions"])
    if args.drop_note_prefix:
        pf_t = np.asarray(obs.memory_v5_prefill_tokens); pf_m = np.asarray(obs.memory_v5_prefill_mask).copy()
        for r in range(pf_t.shape[1]):
            if pf_m[0, r].any():
                txt = video._decode_text(sp, pf_t[0, r][pf_m[0, r]])
                if any(pre == 'ALL' or txt.startswith(pre.replace('_', ' ')) for pre in args.drop_note_prefix):
                    pf_m[0, r] = False; print('dropped prefill row:', repr(txt), flush=True)
        obs = obs.replace(memory_v5_prefill_mask=jnp.asarray(pf_m))
    if args.blank_images:
        obs = obs.replace(images={k: jnp.zeros_like(v) for k, v in obs.images.items()})
        print('images blanked (zeros) for the whole window', flush=True)
    if args.zero_state:
        obs = obs.replace(state=jnp.zeros_like(obs.state)); print('state zeroed for the whole window', flush=True)
    step_i = first_t + args.step
    causal = np.asarray(obs.tokenized_causal)  # [1, T, L]
    causal_mask = np.asarray(obs.tokenized_causal_mask)
    fast_mask = np.asarray(obs.causal_fast_mask)
    text_mask = causal_mask & ~fast_mask
    row, tmask = causal[0, step_i], text_mask[0, step_i]
    print("step", step_i, "label sentence:", repr(video._decode_text(sp, row[tmask])), flush=True)
    assert battery.go_step_mask(causal, text_mask)[0, step_i], "the chosen step is not a go step"
    positions = battery.find_count_positions(row, tmask)
    digit_index = positions[0] + 1
    n_text = int(tmask.sum())
    print(f"digit at position {digit_index}; {n_text} sentence tokens; prefill rows valid:",
          int(np.asarray(obs.memory_v5_prefill_mask)[0].any(-1).sum()), "pending valid:", bool(np.asarray(obs.memory_v5_pending_mask)[0].any()), flush=True)

    # ---- Path A: sequence loss on count variants (full sentence, and masked after the digit) ----
    sequence_loss = nnx_utils.module_jit(model._compute_sequence_loss_v32, static_argnames=("train", "v4_intervention"))  # noqa: SLF001
    rng = jax.random.key(0)
    go_steps = battery.go_step_mask(causal, text_mask)
    results_a = {}
    for kind in ("full", "digit_only"):
        for count in (1, 2, 3):
            variant = causal.copy()
            vmask = causal_mask.copy()
            for t in range(causal.shape[1]):
                if go_steps[0, t]:
                    variant[0, t] = battery.set_count(causal[0, t], text_mask[0, t], count)
            if kind == "digit_only":
                keep = np.arange(causal.shape[-1]) <= digit_index
                vmask[0, step_i] = causal_mask[0, step_i] & (keep | fast_mask[0, step_i])
            losses = sequence_loss(rng, obs.replace(tokenized_causal=jnp.asarray(variant), tokenized_causal_mask=jnp.asarray(vmask)), actions, train=False)
            ce_steps = np.asarray(jax.device_get(losses["v5_step_ce_lm_steps"]))  # [T, b] or [b, T]
            ce = float(ce_steps[step_i, 0] if ce_steps.shape[0] == causal.shape[1] else ce_steps[0, step_i])
            ntok = n_text if kind == "full" else int((text_mask[0, step_i] & (np.arange(causal.shape[-1]) <= digit_index)).sum())
            results_a[(kind, count)] = (ce, ce * ntok)
            print(f"  path A {kind:10s} count={count}: step CE mean {ce:.4f}  sum over {ntok} tokens {ce * ntok:.3f}", flush=True)

    # ---- Path B: rollout machinery, same bank (prefill rows at rate 1 with gaps, pending as prev) ----
    s_len = cfg.model.memory_v5_sentence_len
    sem = model.memory_semantic.init_state(1)
    prev = np.full((1, s_len), -1, dtype=np.int32)
    pf_tokens = np.asarray(obs.memory_v5_prefill_tokens)[0]
    pf_mask = np.asarray(obs.memory_v5_prefill_mask)[0]
    pf_gaps = np.asarray(obs.memory_v5_prefill_gaps)[0]
    notes = []
    for r in range(pf_tokens.shape[0]):
        rm = pf_mask[r, :s_len]
        if not rm.any():
            continue
        rt = np.where(rm, pf_tokens[r, :s_len], 0).astype(np.int32)
        sem, _aux, _keys = model.v6_semantic_write_tokens(sem, jnp.asarray(rt[None]), jnp.asarray(rm[None]), jnp.asarray([True]), rate=1.0)
        sem, _ = model.memory_semantic.analytic_decay(sem, jnp.asarray([max(int(pf_gaps[r]), 0)], dtype=jnp.int32))
        prev = rt[None]
        notes.append(video._decode_text(sp, rt[rm]))
    pend_mask = np.asarray(obs.memory_v5_pending_mask)[0, :s_len]
    if pend_mask.any() and getattr(cfg.model, "memory_v5_write_delay_steps", 0) == 0:
        prev = np.where(pend_mask, np.asarray(obs.memory_v5_pending_tokens)[0, :s_len], 0).astype(np.int32)[None]
        notes.append("(pending) " + video._decode_text(sp, prev[0][pend_mask]))
    print("  path B bank from the window prefill:", notes, flush=True)
    step_obs = _model.Observation(
        images={k: jnp.asarray(v[:, step_i]) for k, v in obs.images.items()},
        image_masks={k: jnp.asarray(v[:, step_i]) for k, v in obs.image_masks.items()},
        state=jnp.asarray(obs.state[:, step_i]),
        tokenized_prompt=jnp.asarray(obs.tokenized_prompt[:, step_i]),
        tokenized_prompt_mask=jnp.asarray(obs.tokenized_prompt_mask[:, step_i]),
    )
    q_tokens, q_mask = jnp.asarray(np.maximum(prev, 0), dtype=jnp.int32), jnp.asarray(prev > 0)
    decode = video.make_decode_fn(model, 24)
    gen_tokens, gen_mask, gen_prob, _, _ = decode(model, step_obs, jnp.asarray(obs.token_state_mask[:, step_i]), sem, q_tokens, q_mask)
    gen_tokens, gen_mask, gen_prob = (np.asarray(x)[0] for x in (gen_tokens, gen_mask, gen_prob))
    print(f"  path B greedy decode: {video._decode_text(sp, gen_tokens[gen_mask])!r} lowest token prob {float(gen_prob[gen_mask].min()):.3f}", flush=True)
    score = make_score_fn(model)
    for count in (1, 2, 3):
        vrow = battery.set_count(row, tmask, count)
        forced = np.where(tmask, vrow, 0).astype(np.int32)[:s_len][None]
        fmask = tmask[:s_len][None]
        logp, dd = score(model, step_obs, sem, q_tokens, q_mask, jnp.asarray(forced), jnp.asarray(fmask))
        logp, dd = np.asarray(logp)[0], np.asarray(dd)[0]
        span = np.where(fmask[0])[0]
        total = -float(logp[span].sum())
        upto = -float(logp[span[span <= digit_index]].sum())
        probs = np.exp(dd[digit_index])
        per_tok = " ".join(f"{sp.id_to_piece(int(forced[0, j]))!s}:{-logp[j]:.2f}" for j in span)
        print(f"  path B count={count}: -log p sum {total:.3f} (prefix+digit {upto:.3f}); digit p(1,2,3) at the digit position = {probs.round(3).tolist()}", flush=True)
        print(f"      per token: {per_tok}", flush=True)
    a = results_a
    print("SUMMARY: path A ranks (full)      :", sorted((1, 2, 3), key=lambda c: a[('full', c)][0]),
          "| path A ranks (digit only):", sorted((1, 2, 3), key=lambda c: a[('digit_only', c)][0]), flush=True)


if __name__ == "__main__":
    main()
