"""Held-out episode rollout video for a v5 checkpoint (cluster_v5/README.md §6).

Walks one episode at the training stride with the semantic bank carried across steps, greedily
decodes the subtask sentence at every step (memory read exactly as in training: layer-8 split,
both banks, blind memory rows), applies the v5 write rule to the decoded sentence ("self" mode:
write iff the sentence changed and its mean token probability >= memory_v5_write_conf; "oracle"
mode: write the label sentence whenever it changes, as in stage A/A2 training), and renders the
raw top-camera video with the ground-truth phase sentence, the training target, the decoded
sentence and the bank contents overlaid. H.264 via ffmpeg. Actions are not sampled.

  python scripts/v5_heldout_video.py --config-name pi05_yam_mem_v5_stageA2 --params <ckpt>/params \\
      --episode-index 2 --write-mode self --output-dir <dir>
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import subprocess
import textwrap
import time

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.project_paths as project_paths
import openpi.training.config as _config
import openpi.training.data_loader as data_loader_lib

PALIGEMMA_EOS_TOKEN = 1
STOP_TOKEN = 108  # "\n" — the trained sentence terminator (FASTSubtaskTokenizer.tokenize_split)


@dataclasses.dataclass
class StepRecord:
    step: int
    frame: int
    gt_now: str
    gt_target: str
    pred: str
    conf: float
    changed: bool
    written: bool
    decision: bool
    evidence: bool
    bank: list[str]
    sem_read_rms: float
    qk_cos_max: float
    retracted: bool = False  # v7: this step erased the newest note (flip-back rule) instead of writing


def _decode_text(sp, tokens):
    ids = [int(t) for t in tokens]
    while ids and ids[-1] in (STOP_TOKEN, PALIGEMMA_EOS_TOKEN, 0):
        ids.pop()
    return sp.decode(ids).strip()


def make_decode_fn(model, max_decode_steps: int):
    """One rollout step: read both banks (visual bank blank; injection follows the config), greedily
    decode the subtask sentence against the memory-extended cache, return tokens / per-token probs."""

    @nnx.jit
    def decode(model, observation, state_token_mask, sem_state, prev_tokens, prev_mask):
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        num_img = prefix_len - model.max_token_len
        top_tokens = model._top_camera_token_count(num_img, preprocessed.images)  # noqa: SLF001
        mem_len = model._memory_token_total  # noqa: SLF001
        gen_base = prefix_len + mem_len
        if getattr(model, "memory_v0920_input_read", False):
            # 0920 structure (beans0922_v1 / robomme_0920): the 8 read tokens sit at the INPUT of all blocks, the same
            # dispatch as Pi0._sample_with_memory_v32 and the training scan (no layer-8 interface, no visual bank).
            prepared = model._v0920_prepare_prefix(  # noqa: SLF001
                prefix_tokens,
                prefix_mask,
                prefix_ar,
                sem_state,
                top_token_count=top_tokens,
                visual_state=model.memory.init_state(batch),
                state=preprocessed.state,
                prev_tokens=prev_tokens,
                prev_mask=prev_mask,
            )
        else:
            prepared = model._v32_prepare_memory_prefix(  # noqa: SLF001
                prefix_tokens,
                prefix_mask,
                prefix_ar,
                model.memory.init_state(batch),
                top_token_count=top_tokens,
                state_token_mask=state_token_mask,
                semantic_state=sem_state,
                v5_prev_tokens=prev_tokens,
                v5_prev_mask=prev_mask,
            )
        kv_cache = prepared["cache"]
        final_prefix = prepared["final_prefix"]
        memory_valid = prepared["memory_valid"]
        causal_len = model.causal_token_len

        pointer_on = getattr(model, "memory_v6_pointer_read", False) and sem_state is not None

        context_pointer = pointer_on and getattr(model, "memory_v6_pointer_query", "hidden") == "context"

        def logits_of(hidden_vec, index, so_far_tokens=None, so_far_mask=None):
            logits = model.PaliGemma.llm(hidden_vec[:, None], method="decode")[:, 0].astype(jnp.float32)
            if pointer_on:  # v6 pointer read on the sentence positions (same rule as Pi0._sample_with_memory_v32)
                on_span = jnp.broadcast_to(jnp.asarray(index) < model.memory_v5_sentence_len, (batch, 1))
                queries = None
                if context_pointer:
                    s_len = model.memory_v5_sentence_len
                    if so_far_tokens is None:
                        queries = jnp.zeros((batch, 1, model.memory_semantic.config.d_key), dtype=jnp.float32)
                    else:
                        ctx_q = model.v6_context_queries(so_far_tokens[:, :s_len], so_far_mask[:, :s_len])
                        idx = jnp.clip(jnp.asarray(index), 0, s_len - 1)
                        queries = jax.lax.dynamic_index_in_dim(ctx_q, idx, axis=1, keepdims=True)
                logits = logits + model.v6_pointer_bonus(
                    hidden_vec[:, None], sem_state, on_span, logits.shape[-1], queries=queries
                )[:, 0]
            return logits

        def pick(logits):
            probs = jax.nn.softmax(logits, axis=-1)
            token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            return token, jnp.take_along_axis(probs, token[:, None], axis=-1)[:, 0]

        token0, prob0 = pick(logits_of(model._v32_causal_seed(final_prefix, prefix_mask)[:, 0], 0))  # noqa: SLF001
        gen_tokens = jnp.zeros((batch, causal_len), dtype=jnp.int32)
        gen_mask = jnp.zeros((batch, causal_len), dtype=bool)
        gen_prob = jnp.zeros((batch, causal_len), dtype=jnp.float32)

        def record(tokens, mask, prob, done, token, p, index):
            tokens = tokens.at[:, index].set(jnp.where(done, tokens[:, index], token))
            mask = mask.at[:, index].set(~done)
            prob = prob.at[:, index].set(jnp.where(done, prob[:, index], p))
            return tokens, mask, prob, done | (token == STOP_TOKEN) | (token == PALIGEMMA_EOS_TOKEN)

        gen_tokens, gen_mask, gen_prob, done = record(
            gen_tokens, gen_mask, gen_prob, jnp.zeros(batch, dtype=bool), token0, prob0, 0
        )

        def cond(carry):
            return (carry[-1] < max_decode_steps) & ~jnp.all(carry[3])

        def step(carry):
            tokens, mask, prob, done, previous, cache, index = carry
            token_emb = model.PaliGemma.llm(previous[:, None], method="embed")
            step_attn = model._v32_step_mask(prefix_mask, index, memory_valid=memory_valid)  # noqa: SLF001
            patterns = getattr(model, "memory_v7_digit_blind_patterns", ())
            if patterns:  # v7 digit blinding, same rule as Pi0._sample_with_memory_v32
                rows = _pi0.digit_blind_rows(tokens, patterns)
                row = jnp.take_along_axis(rows, jnp.broadcast_to(index - 1, (batch, 1)), axis=1)
                own = jnp.broadcast_to(gen_base + index - 1, (batch, 1))
                step_attn = model._v7_digit_blind(step_attn, row, own, prefix_len)  # noqa: SLF001
            (out, _), cache = model.PaliGemma.llm(
                [token_emb, None],
                mask=step_attn,
                positions=jnp.broadcast_to(gen_base + index - 1, (batch, 1)),
                kv_cache=cache,
                cache_position=gen_base + index - 1,
            )
            token, p = pick(logits_of(out[:, 0], index, tokens, mask))
            tokens, mask, prob, done = record(tokens, mask, prob, done, token, p, index)
            return tokens, mask, prob, done, token, cache, index + 1

        carry = (gen_tokens, gen_mask, gen_prob, done, token0, kv_cache, jnp.asarray(1, dtype=jnp.int32))
        gen_tokens, gen_mask, gen_prob, _, _, _, _ = jax.lax.while_loop(cond, step, carry)
        sem_rms = jnp.sqrt(jnp.mean(jnp.square(prepared["sem_retrieved"].astype(jnp.float32)), axis=(1, 2)))
        return gen_tokens, gen_mask, gen_prob, sem_rms, prepared.get("sem_queries")

    return decode


def make_write_fn(model):
    @nnx.jit
    def write(model, tokens, mask, sem_state, commit):
        if getattr(model, "memory_v6_token_writes", False):
            # v6: token-level write; the diagnostic key is the mean token key
            new_state, aux, pooled = model.v6_semantic_write_tokens(sem_state, tokens, mask, commit)
            return new_state, jnp.any(aux["commit_applied"], axis=-1), pooled[:, 0]
        keys, values = model.v5_sentence_kv(tokens, mask)  # A8-aware (== encode+intent without the flags)
        new_state, aux = model.v5_semantic_write(sem_state, keys, values, commit)
        return new_state, aux["commit_applied"][:, 0], keys[:, 0]

    return write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_yam_mem_v5_stageA2")
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True, help="LeRobot episode index (manifest episode_index)")
    parser.add_argument("--write-retry", action="store_true",
                        help="retry-until-committed change detector (prev = last COMMITTED sentence); default = the config's memory_v5_prev_is_committed")
    parser.add_argument("--write-mode", choices=("self", "self_nodup", "self_stable", "self_debounce", "oracle", "oracle_evidence"), default="self",
                        help="self: own decoded sentences; oracle: every label change; oracle_evidence: labels only for the "
                             "frames BEFORE the closing segment (the notes), own sentences from the closing segment on -- the "
                             "recall test: with correct notes in the bank, does the model restate the target's note and decide?")
    parser.add_argument(
        "--intervention",
        choices=("none", "flip_sides", "blank", "freeze", "freeze_decision"),
        default="none",
        help="flip_sides: swap the side words (left<->right) in every sentence WRITTEN to the bank; "
        "blank: never commit (the semantic bank stays empty). Decode targets/overlays are unchanged. "
        "freeze: feed the FIRST step's observation (images, state; LED off, arm still) at every step, so the only "
        "thing that advances is the memory — a counter that still reports blinks runs on a timing prior, not on "
        "the LED (2026-09-06 20:30, real-robot report: ckpt 2750 counts blinks with the LED dark). "
        "freeze_decision: hold the last pre-decision frame/state through the decision segment (no arm-motion cue).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                        help="episode manifest (default: the frozen v36 bins manifest); v5 generic manifests work too")
    parser.add_argument("--sidecar", type=pathlib.Path, default=None,
                        help="v5 sentence sidecar (default: the bins sidecar)")
    parser.add_argument("--debounce-steps", type=int, default=2,
                        help="self_stable / self_debounce: commit a sentence only after it was produced on this many consecutive steps")
    parser.add_argument("--max-decode-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--set-param", action="append", default=[], metavar="PATH=VALUE",
                        help="overwrite a scalar leaf of the restored params before loading, e.g. "
                        "memory_v6_pointer_beta/value=10 (counterfactual: r3's beta never recovered from its reset)")
    parser.add_argument("--tag-suffix", default="", help="appended to the output tag (json/mp4 names)")
    parser.add_argument("--retract-steps", type=int, default=None,
                        help="v7 flip-back retraction window in self modes (default: the config's memory_v7_write_retract_steps)")
    parser.add_argument("--vocab-only", type=int, default=None,
                        help="1/0: write only complete reference sentences in self modes (default: the config's memory_v7_write_vocab_only)")
    parser.add_argument("--stride", type=int, default=0,
                        help="override the memory stride in FRAMES (0 = the config's memory_stride_frames, 5 for the beans "
                        "models = 167 ms ticks at 30 Hz). 8 emulates the robot client at --hz 20 with a replan every 5 "
                        "controls (250 ms ticks): the LED cue then spans 1.6x fewer memory steps than in training "
                        "(2026-09-06 21:00, real-robot report: ckpt 2750 sometimes opens with 'light on: 2 green blinks').")
    parser.add_argument("--write-conf", type=float, default=None,
                        help="override the checkpoint's memory_v5_write_conf threshold (the gate's rule, mean or lowest word, is the config's)")
    parser.add_argument("--pointer-beta", type=float, default=None,
                        help="diagnostic (0920_v4): override the trained pointer scale memory_v6_pointer_beta (0 = no pointer bonus)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = _config.get_config(args.config_name)
    params = _model.restore_params(args.params, restore_type=np.ndarray)
    for spec in args.set_param:
        path, val = spec.split("=", 1); node = params
        keys = [k for k in path.split("/") if k != "value"]  # restore_params strips nnx's trailing "value"
        if keys[0] not in node and "params" in node: node = node["params"]
        for k in keys[:-1]: node = node[k]
        old = node[keys[-1]]; node[keys[-1]] = np.asarray(float(val), dtype=old.dtype).reshape(np.shape(old))
        print(f"set-param {path}: {old} -> {node[keys[-1]]}")
    model = cfg.model.load(params)
    model.eval()
    if args.pointer_beta is not None and hasattr(model, "memory_v6_pointer_beta"):
        model.memory_v6_pointer_beta.value = jnp.asarray(float(args.pointer_beta), dtype=jnp.float32)
        print(f"pointer scale override: memory_v6_pointer_beta = {args.pointer_beta}", flush=True)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if args.stride > 0 and args.stride != data_config.memory_stride_frames:
        # Every stride read happens inside create_torch_dataset (window offsets, MemorySequenceSubtasks, the
        # episode/sampling tables), so a post-create replace reaches the whole pipeline.
        print(f"memory stride override: {data_config.memory_stride_frames} -> {args.stride} frames", flush=True)
        data_config = dataclasses.replace(data_config, memory_stride_frames=args.stride)
    dataset = data_loader_lib.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = data_loader_lib.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, *data_config.model_transforms.inputs],
    )
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )
    manifest_path = args.manifest or project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = args.sidecar or project_paths.project_path(project_paths.V5_SUBTASK_LABELS)
    sidecar = json.loads(sidecar_path.read_text())
    episodes = sorted([e for e in manifest["episodes"] if e.get("include", True)], key=lambda e: e["episode_index"])
    episode = episodes[args.episode_index]
    assert episode["episode_index"] == args.episode_index
    start = sum(int(e["expected_num_frames"]) for e in episodes[: args.episode_index])
    length = int(episode["expected_num_frames"])
    segments = sidecar["episodes"][episode["stable_id"]]["segments"]
    frame_sentence = np.empty(length, dtype=object)
    for seg in segments:
        frame_sentence[seg["start"] : seg["end"] + 1] = seg["sentence"]
    if "raw_dir" in episode:
        raw_dir = pathlib.Path(episode["raw_dir"])
        if not raw_dir.is_absolute():
            raw_root = pathlib.Path(manifest.get("raw_root", "."))
            if not raw_root.is_absolute():
                raw_root = manifest_path.parent / raw_root
            raw_dir = raw_root / raw_dir
        video_path = (raw_dir / "top_camera_rgb.mp4").resolve()
    else:
        # RoboMME (cluster_robomme/README.md): no raw demo directory; the front camera is the LeRobot "image" video
        # of the configured dataset (lossless H264, one frame per dataset row), the prompt is setup/task_goal[0].
        video_path = (
            pathlib.Path(data_config.lerobot_dataset_root) / "videos" / f"chunk-{args.episode_index // 1000:03d}"
            / "image" / f"episode_{args.episode_index:06d}.mp4"
        ).resolve()
    episode_prompt = episode.get("prompt") or (episode.get("task_goals") or [""])[0]
    # bins: target_side names the answer; generic manifests carry a class label instead (e.g. "x=3").
    episode_target = episode.get("target_side") or episode.get("class") or ""
    print(f"episode {args.episode_index} {episode['stable_id']} prompt={episode_prompt!r} target={episode_target} "
          f"frames={length} start={start} video={video_path} (setup {time.time() - t0:.0f}s)", flush=True)

    stride = data_config.memory_stride_frames
    steps_per_window = cfg.model.memory_seq_steps
    lookahead = data_config.subtask_lookahead
    sentence_len = cfg.model.memory_v5_sentence_len
    conf_threshold = cfg.model.memory_v5_write_conf if args.write_conf is None else float(args.write_conf)
    conf_use_min = bool(getattr(cfg.model, "memory_v5_write_conf_min", False))
    # oracle_evidence: label writes stop at the closing segment (second-to-last sidecar segment = the restated note)
    # oracle_evidence: label notes are handed over up to the closing note; with a merged tail (sidecar "tail_merged":
    # closing + decision = one segment) that is the tail segment itself
    _evidence_end = segments[-1]["start"] if sidecar.get("tail_merged") else (segments[-2]["start"] if len(segments) >= 2 else length)
    oracle_until = int(_evidence_end) if args.write_mode == "oracle_evidence" else length
    prev_is_committed = bool(args.write_retry or getattr(cfg.model, "memory_v5_prev_is_committed", False))
    decode = make_decode_fn(model, args.max_decode_steps)
    write = make_write_fn(model)

    sem_state = model.memory_semantic.init_state(1)
    prev_tokens = np.full((1, sentence_len), -1, dtype=np.int32)
    pending = (np.zeros((1, sentence_len), dtype=np.int32), np.zeros(sentence_len, dtype=bool), False)
    bank: list[str] = []
    bank_keys: list[np.ndarray] = []
    last_cand_text = None  # self_stable: the previous step's candidate sentence (debounce)
    cand_streak = 0  # consecutive steps that produced the same candidate (--debounce-steps)
    grammar_rejections = 0  # v7 phase-grammar gate: candidates refused because their phase cannot follow the bank's newest
    # v7 (09-17) generic write rules, same as the training scan: vocabulary-only writes and flip-back retraction
    retract_k = int(getattr(cfg.model, "memory_v7_write_retract_steps", 0)) if args.retract_steps is None else int(args.retract_steps)
    vocab_only = bool(getattr(cfg.model, "memory_v7_write_vocab_only", False)) if args.vocab_only is None else bool(args.vocab_only)
    ref_rows = {tuple(int(t) for t in row) for row in cfg.model.memory_v5_reference_tokens}
    sem_out_name = model.memory_semantic._output_weight_name
    rho_one = float(1.0 - model.memory_semantic.config.alpha_step)
    last_delta = None  # w3 delta of the newest committed note
    commit_age = 10**6  # steps since the newest note entered (huge = nothing retractable)
    committed_tok = []  # token rows of the notes in `bank` (parallel list; every-step writes keep it unused)
    retractions = 0
    vocab_rejections = 0
    records: list[StepRecord] = []
    frozen = None  # --intervention freeze: the first step's (observation, state_token_mask)
    last_still = None  # --intervention freeze_decision: the last pre-decision step's (observation, state_token_mask)
    step_index = 0
    window_start = 0
    while window_start < length:
        # The training transform refuses a window whose decision steps have no evidence anchor
        # inside it (a D phase straddling a window boundary).  For the rollout we simply fetch
        # the window from an earlier frame so the anchor is included, and skip the steps already
        # processed; the memory state itself is carried step by step and is unaffected.
        first_t = 0
        while True:
            fetch_start = window_start - first_t * stride
            try:
                item = tds[start + fetch_start]
                break
            except ValueError as err:
                if "E anchor" not in str(err) or fetch_start - stride < 0:
                    raise
                first_t += 1
        if first_t:
            print(f"window at frame {window_start} fetched from frame {fetch_start} (skipping {first_t} steps)", flush=True)
        window_start = fetch_start
        batched = jax.tree.map(lambda x: np.asarray(x)[None], item)
        # The same conversion the training loader applies (uint8 images -> [-1, 1] float, field mapping).
        seq_obs = _model.Observation.from_dict(batched)
        step_mask = np.asarray(seq_obs.seq_step_mask)[0]
        decision_mask = np.asarray(seq_obs.seq_decision_mask)[0]
        write_mask = np.asarray(seq_obs.seq_write_mask)[0]
        causal = np.asarray(seq_obs.tokenized_causal)[0]
        causal_mask = np.asarray(seq_obs.tokenized_causal_mask)[0]
        causal_fast = np.asarray(seq_obs.causal_fast_mask)[0]
        next_start = window_start + steps_per_window * stride
        for t in range(first_t, steps_per_window):
            frame = window_start + t * stride
            if not step_mask[t] or frame >= length:
                break
            # One time slice, exactly as the training scan builds its per-step observation.
            observation = _model.Observation(
                images={k: jnp.asarray(v[:, t]) for k, v in seq_obs.images.items()},
                image_masks={k: jnp.asarray(v[:, t]) for k, v in seq_obs.image_masks.items()},
                state=jnp.asarray(seq_obs.state[:, t]),
                tokenized_prompt=jnp.asarray(seq_obs.tokenized_prompt[:, t]),
                tokenized_prompt_mask=jnp.asarray(seq_obs.tokenized_prompt_mask[:, t]),
            )
            state_token_mask = jnp.asarray(seq_obs.token_state_mask[:, t])
            if args.intervention == "freeze":
                if frozen is None:
                    frozen = (observation, state_token_mask)
                observation, state_token_mask = frozen
            if args.intervention == "freeze_decision":
                # 2026-09-09 05:25: hold the LAST pre-decision observation (lids closed, arm still) through the whole
                # decision segment, so the decoded `open bin k` cannot read the robot's own motion from the images or
                # the joint state; the human phase is seen normally (notes can be perceived/written).
                if bool(decision_mask[t]) and last_still is not None:
                    observation, state_token_mask = last_still
                elif not bool(decision_mask[t]):
                    last_still = (observation, state_token_mask)
            # A6: the last decoded sentence (the delay's pending sentence) conditions the read queries.
            # The read queries condition on the previous sentence exactly as the training scan does:
            # delay 1 -> the pending (one-step-delayed) sentence; delay 0 -> the previous sentence
            # (prev_tokens: last produced, or last committed under the retry rule). Before 2026-09-05
            # 12:40 the delay-0 rollout always passed the never-filled pending slot (an empty sentence).
            if getattr(cfg.model, "memory_v5_write_delay_steps", 0) == 1:
                q_tokens, q_mask = pending[0], pending[1][None]
            else:
                q_tokens, q_mask = np.maximum(prev_tokens, 0), prev_tokens > 0
            gen_tokens, gen_mask, gen_prob, sem_rms, sem_queries = decode(
                model, observation, state_token_mask, sem_state,
                jnp.asarray(q_tokens, dtype=jnp.int32), jnp.asarray(q_mask),
            )
            gen_tokens = np.asarray(gen_tokens)[0]
            gen_mask = np.asarray(gen_mask)[0]
            gen_prob = np.asarray(gen_prob)[0]
            pred = _decode_text(sp, gen_tokens[gen_mask])
            if gen_mask.any():
                # the gate's sentence confidence: the mean token probability, or (memory_v5_write_conf_min, 0920_v4) the lowest
                conf = float(gen_prob[gen_mask].min() if conf_use_min else gen_prob[gen_mask].mean())
            else:
                conf = 0.0
            # the sentence to (maybe) write
            oracle_here = args.write_mode == "oracle" or (args.write_mode == "oracle_evidence" and frame + lookahead < oracle_until)
            if oracle_here:
                span = (causal_mask[t] & ~causal_fast[t])[:sentence_len]
                cur = np.where(span, causal[t][:sentence_len], 0).astype(np.int32)[None]
                confident = True
                # A3 training protocol: a waiting label is stored side-stripped ("wait\n").
                prefix = tuple(cfg.model.memory_v5_bank_waiting_prefix)
                if prefix and span[: len(prefix)].all() and tuple(cur[0, : len(prefix)].tolist()) == prefix:
                    tokens = tuple(cfg.model.memory_v5_bank_waiting_tokens)
                    cur = np.zeros((1, sentence_len), dtype=np.int32)
                    cur[0, : len(tokens)] = tokens
                    span = np.arange(sentence_len) < len(tokens)
            else:
                span = np.zeros(sentence_len, dtype=bool)
                n = int(min(gen_mask.sum(), sentence_len))
                span[:n] = True
                cur = np.zeros((1, sentence_len), dtype=np.int32)
                cur[0, :n] = gen_tokens[gen_mask][:n]
                confident = conf >= conf_threshold
            if getattr(cfg.model, "memory_v5_write_delay_steps", 0) == 1:
                # A4: write what was produced one step ago (nothing at the first step).
                produced = (cur.copy(), span.copy(), confident)
                cur, span, confident = pending
                pending = produced
            if args.intervention == "flip_sides":
                # PaliGemma ids: 2731 = " left", 1833 = " right" (the sidecar's side words).
                flipped = np.where(cur == 2731, 1833, np.where(cur == 1833, 2731, cur))
                cur = flipped.astype(np.int32)
            write_every = bool(getattr(cfg.model, "memory_v7_write_every_step", False))  # v7 gradual bank
            changed = (bool(np.any(cur != prev_tokens)) or write_every) and bool(span.any())
            commit = changed and confident and args.intervention != "blank"
            # self_nodup (2026-09-12 23:25, B/500 battery): every boba sentence occurs once per episode, so a
            # candidate that is already in the bank is a transient regression to an old sentence (the
            # self-write count drift: 'scoop 1 of 3' re-committed in the middle of scoop 2); do not commit it.
            cand_text = _decode_text(sp, cur[0][span]) if span.any() else ""
            if commit and args.write_mode in ("self_nodup", "self_stable") and cand_text in bank:
                commit = False
            # self_stable = self_nodup + debounce: commit only a sentence produced on two consecutive steps (the
            # empty-bank start flickers 'first, place cup' / 'press tap' for single steps before 'watch, sago').
            # self_debounce (RoboMME 09-15): the debounce alone, duplicates allowed -- the official PickXtimes place
            # sentence legitimately repeats every cycle, which self_stable's no-duplicate rule blocks.
            cand_streak = cand_streak + 1 if cand_text == last_cand_text else 1
            if commit and args.write_mode in ("self_stable", "self_debounce") and cand_streak < args.debounce_steps:
                commit = False
            # v7: a model trained with a debounce applies the same rule in plain self mode
            if commit and args.write_mode == "self" and cand_streak < int(getattr(cfg.model, "memory_v7_write_debounce_steps", 1)):
                commit = False
            # v7 phase-grammar gate (09-16): the candidate's first token must be allowed to follow the newest COMMITTED
            # sentence's first token (prev_tokens; -1 = empty bank -> the labels' initial phases). Same rule as the
            # training scan (Pi0.v7_grammar_allows); rejected candidates are retried at the next steps.
            grammar = tuple(getattr(cfg.model, "memory_v7_write_grammar", ()))
            if commit and args.write_mode == "self" and grammar:
                prev_first, cur_first = int(prev_tokens[0, 0]), int(cur[0, 0])
                allowed = (cur_first in tuple(getattr(cfg.model, "memory_v7_write_grammar_initial", ()))) if prev_first < 0 \
                    else ((prev_first, cur_first) in {(int(a), int(b)) for a, b in grammar})
                if not allowed:
                    commit = False
                    grammar_rejections += 1
            last_cand_text = cand_text
            retracted_now = False
            if commit and args.write_mode.startswith("self") and vocab_only and tuple(int(t) for t in cur[0][span]) not in ref_rows:
                commit = False
                vocab_rejections += 1
            if (commit and args.write_mode.startswith("self") and retract_k > 0 and not write_every and last_delta is not None
                    and len(committed_tok) >= 2 and commit_age + 1 <= retract_k and np.array_equal(cur, committed_tok[-2])):
                # A -> B -> A within retract_k steps: erase B instead of writing A again (training scan rule)
                commit = False
                retracted_now = True
            w3_before = np.asarray(sem_state.fast_weights[sem_out_name], dtype=np.float32)
            sem_state, applied, key = write(model, jnp.asarray(cur), jnp.asarray(span[None]), sem_state, jnp.asarray([commit]))
            applied = bool(np.asarray(applied)[0])
            if retracted_now:
                w3 = np.asarray(sem_state.fast_weights[sem_out_name], dtype=np.float32) - (rho_one ** (commit_age + 1)) * last_delta
                sem_state = model.memory_semantic._canonical_delta_state(sem_state, jnp.asarray(w3))
                bank.pop(); bank_keys.pop(); committed_tok.pop()
                last_delta = None; commit_age = 10**6; retractions += 1
            elif applied and not write_every:
                last_delta = np.asarray(sem_state.fast_weights[sem_out_name], dtype=np.float32) - rho_one * w3_before
                commit_age = 0
                committed_tok.append(cur.copy())
            else:
                commit_age = min(commit_age + 1, 10**6)
            if applied or not prev_is_committed:
                prev_tokens = cur
            if retracted_now:
                prev_tokens = committed_tok[-1].copy()  # the note before B is the newest again
            written_text = _decode_text(sp, cur[0][span]) if commit else ""
            if applied and not (write_every and bank and bank[-1] == written_text):  # every-step writes: list distinct runs
                bank.append(written_text)
                bank_keys.append(np.asarray(key)[0])
            qk = 0.0
            if sem_queries is not None and bank_keys:
                q = np.asarray(sem_queries)[0]
                qk = float(np.max(q @ np.stack(bank_keys).T))
            gt_now = str(frame_sentence[frame])
            gt_target = str(frame_sentence[min(frame + lookahead, length - 1)])
            records.append(
                StepRecord(step_index, frame, gt_now, gt_target, pred, conf, changed, applied, bool(decision_mask[t]),
                           bool(write_mask[t]), list(bank), float(np.asarray(sem_rms)[0]), qk, retracted=retracted_now)
            )
            flag = "W" if applied else ("R" if retracted_now else " ")
            d = "D" if decision_mask[t] else " "
            print(f"[{step_index:3d} f{frame:4d} {d}{flag}] pred={pred!r} conf={conf:.2f} | target={gt_target!r} | bank={len(bank)}", flush=True)
            step_index += 1
        window_start = next_start

    # ---- summary
    decisions = [r for r in records if r.decision]
    def side(s):
        return "left" if " left" in f" {s}" else "right" if " right" in f" {s}" else None
    first = decisions[0] if decisions else None
    target_side = episode.get("target_side")
    # Generic tasks (no side words): a decision step is correct when the decoded sentence equals
    # its lookahead-shifted label exactly; bins keep the side-word criterion.
    def decision_ok(r):
        return (side(r.pred) == target_side) if target_side else (r.pred == r.gt_target)
    summary = {
        "episode_index": args.episode_index,
        "stable_id": episode["stable_id"],
        "prompt": episode_prompt,
        "target_side": target_side,
        "target": episode_target,
        "write_mode": args.write_mode,
        "intervention": args.intervention,
        "steps": len(records),
        "decision_steps": len(decisions),
        "decision_side_correct": sum(1 for r in decisions if decision_ok(r)),
        "decision_exact": sum(1 for r in decisions if r.pred == r.gt_target),
        "first_decision_pred": first.pred if first else None,
        "first_decision_correct": decision_ok(first) if first else None,
        "evidence_pred_exact": sum(1 for r in records if r.evidence and r.pred == r.gt_target),
        "evidence_steps": sum(1 for r in records if r.evidence),
        "writes": sum(1 for r in records if r.written),
        "grammar_rejections": grammar_rejections,
        "retractions": retractions,
        "vocab_rejections": vocab_rejections,
        "final_bank": bank,
        "records": [dataclasses.asdict(r) for r in records],
    }
    tag = f"ep{args.episode_index:02d}_{args.write_mode}" + ("" if args.intervention == "none" else f"_{args.intervention}") \
        + ("" if args.debounce_steps == 2 or args.write_mode not in ("self_stable", "self_debounce") else f"_d{args.debounce_steps}") \
        + ("" if args.stride <= 0 else f"_stride{args.stride}") + args.tag_suffix
    (args.output_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"decision steps {summary['decision_side_correct']}/{summary['decision_steps']} correct "
          f"({'true side' if target_side else 'exact sentence'}); "
          f"first decision: {summary['first_decision_pred']!r} ({'OK' if summary['first_decision_correct'] else 'WRONG'}); "
          f"inspect sentence exact {summary['evidence_pred_exact']}/{summary['evidence_steps']}; writes={summary['writes']}", flush=True)

    # ---- render
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 640:  # RoboMME 256-px frames: upscale 3x so the wrapped captions fit (cluster_robomme/rerender_video.py)
        width, height = width * 3, height * 3
    band = 24 * 9 + 16
    out_path = args.output_dir / f"{tag}.mp4"
    ffmpeg = shutil.which("ffmpeg")
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height + band}",
         "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
         str(out_path)],
        stdin=subprocess.PIPE,
    )
    by_frame = {}
    for r in records:
        by_frame[r.frame] = r
    current = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_id = 0
    while True:
        ok, img = cap.read()
        if not ok or frame_id >= length:
            break
        if frame_id in by_frame:
            current = by_frame[frame_id]
        # RoboMME (09-15): 256-px frames are upscaled so the captions fit, and every caption line wraps (before, the
        # text ran off the right edge and only the first words were visible). Wider frames (YAM) are unchanged.
        canvas = np.zeros((height + band, width, 3), dtype=np.uint8)
        canvas[:height] = img if (width, height) == (img.shape[1], img.shape[0]) else cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        y = height + 22
        def put(text, color=(255, 255, 255), scale=0.55, max_lines=2):
            nonlocal y
            chars = max(20, int(width / (9.6 * scale / 0.55)))
            for line in textwrap.wrap(text, chars)[:max_lines] or [""]:
                cv2.putText(canvas, line, (8, y), font, scale, color, 1, cv2.LINE_AA)
                y += 24
        put(f"{episode['stable_id']}  prompt: {episode_prompt}  frame {frame_id}  [{args.write_mode} writes{'' if args.intervention == 'none' else ' / ' + args.intervention}]", (200, 200, 200), 0.5)
        put(f"GT phase : {frame_sentence[frame_id]}", (255, 255, 255))
        if current is not None:
            ok_pred = current.pred == current.gt_target
            put(f"PRED @{current.frame}: {current.pred}   (conf {current.conf:.2f})", (80, 220, 80) if ok_pred else (60, 60, 255))
            put(f"target   : {current.gt_target}" + ("   DECISION STEP" if current.decision else ""), (180, 180, 180))
            put(f"bank[{len(current.bank)}]: " + (" | ".join(current.bank[-3:]) if current.bank else "(empty)") + ("   <- WRITE" if current.written else ""), (255, 200, 80))
        if current is not None and current.decision and frame_id - current.frame < stride:
            cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (0, 200, 255), 3)
        proc.stdin.write(canvas.tobytes())
        frame_id += 1
    proc.stdin.close()
    proc.wait()
    cap.release()
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {frame_id} frames, H.264) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
