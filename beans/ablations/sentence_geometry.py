"""CPU-only sentence/template or contextual-token geometry and oracle-write recall.

Use --initialize with a stage-A config for the exact training seed/KI graft, or
--params for a trained checkpoint. This is bank/representation diagnostics, not
autoregressive accuracy or a test of the predicted-sentence confidence gate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

# Set before importing JAX: never compete with an ongoing GPU training job.
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np


def cosine_matrix(left, right=None):
    left = np.asarray(left, dtype=np.float64)
    right = left if right is None else np.asarray(right, dtype=np.float64)
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    return left @ right.T


def pair_summary(matrix, mask):
    values = np.asarray(matrix)[np.asarray(mask, dtype=bool)]
    if not len(values):
        return {"pairs": 0}
    return {"pairs": int(len(values)), "min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "mean_abs": float(np.abs(values).mean())}


def template_groups(rows, max_diff):
    from openpi.models.sentence_slots import template_masks, template_representatives

    masks = template_masks(rows, max_diff)
    signatures = [tuple(t for t, keep in zip(row, mask, strict=True) if keep)
                  for row, mask in zip(rows, masks, strict=True)]
    representatives = template_representatives(rows, max_diff)
    addresses = [signatures[i] for i in representatives]
    return representatives, np.asarray([addresses.index(s) for s in signatures])


def load_model(config, args, output):
    import jax
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.training import weight_loaders

    if args.params:
        params = model_lib.restore_params(args.params, restore_type=np.ndarray)
        provenance = {"kind": "trained_checkpoint", "params": str(args.params.resolve()),
                      "parameter_tree_sha256": weight_loaders.parameter_tree_sha256(params)}
        model = config.model.load(params)
    else:
        if not config.model.memory_v5_oracle_writes:
            raise ValueError("--initialize requires the stage-A config; B must load its own trained A.")
        # Exactly train.main -> init_train_state's two RNG splits, graft, and freeze cast.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "openpi/scripts"))
        from train import _cast_frozen_params, _load_weights_and_validate

        init_rng = jax.random.split(jax.random.key(config.seed))[1]
        model_rng = jax.random.split(init_rng)[1]
        abstract = nnx.eval_shape(config.model.create, model_rng)
        graph, shape = nnx.split(abstract)
        loader = dataclasses.replace(config.weight_loader,
                                     manifest_output_path=str(output / "initialization_graft_manifest.json"))
        partial = _load_weights_and_validate(loader, shape.to_pure_dict())

        @jax.jit
        def initialize(rng, graft):
            state = nnx.state(config.model.create(rng))
            state.replace_by_pure_dict(graft)
            return _cast_frozen_params(config, state)

        state = initialize(model_rng, partial)
        model = nnx.merge(graph, state)
        provenance = {"kind": "untrained_exact_stage_a_initialization", "seed": config.seed,
                      "base_params": loader.params_path,
                      "parameter_tree_sha256": weight_loaders.parameter_tree_sha256(state.to_pure_dict())}
    model.eval()
    return model, provenance


def measure(model):
    if getattr(model, "memory_v6_token_writes", False):
        return measure_tokens(model)
    import jax
    import jax.numpy as jnp

    rows = tuple(tuple(row) for row in model.memory_v5_reference_tokens)
    reps, groups = template_groups(rows, model.memory_v5_slot_max_diff)
    tokens, mask = model.v5_reference_token_rows(model.memory_v5_sentence_len)
    # Reuse identical deterministic reference encodings within this read-only probe.
    # Calling the production methods below still computes the actual slot/whitening path.
    encode = model.v5_encode_sentence
    cache = {}

    def cached_encode(t, m):
        key = (np.asarray(t).tobytes(), np.asarray(m).tobytes(), t.shape)
        if key not in cache:
            cache[key] = encode(t, m)
        return cache[key]

    model.v5_encode_sentence = cached_encode
    try:
        encoded = model.v5_encode_sentence(tokens, mask)
        raw_value = model.memory_sem_value_proj(encoded)
        keys, values = model.v5_sentence_kv(tokens, mask)
        if getattr(model, "memory_template_read", False):
            addresses = model.v5_template_keys()
        else:
            # A9 stores the same template-keyed sentences, but its runtime reader
            # is conditioned and has no direct-read memory_template_rows attribute.
            # Probe the real WRITE keys, not invented fixed runtime read queries.
            addresses = keys[jnp.asarray(reps, dtype=jnp.int32), 0]
    finally:
        model.v5_encode_sentence = encode
    keys, values = np.asarray(keys)[:, 0], np.asarray(values)[:, 0]
    addresses = np.asarray(addresses)
    if len(addresses) != len(reps) or not np.allclose(keys, addresses[groups], atol=1e-4):
        raise ValueError("write keys do not match discovered template addresses")
    bank = model.memory_semantic
    blank = bank.init_state(1)
    hidden = np.asarray(bank.hidden_key(blank, jnp.asarray(addresses)[None]))[0]
    matrices = {name: cosine_matrix(vectors) for name, vectors in (
        ("sentence_encoding", encoded), ("raw_projected_value", raw_value),
        ("whitened_write_value", values), ("write_key", keys),
        ("template_key", addresses), ("template_hidden_feature", hidden))}
    upper = np.triu(np.ones((len(rows), len(rows)), dtype=bool), 1)
    summaries = {}
    for name, matrix in matrices.items():
        if name.startswith("template_"):
            summaries[name] = {"different_templates": pair_summary(matrix, np.triu(np.ones_like(matrix, bool), 1))}
        else:
            summaries[name] = {"same_template": pair_summary(matrix, upper & (groups[:, None] == groups[None])),
                               "different_templates": pair_summary(matrix, upper & (groups[:, None] != groups[None]))}

    # No decoder/gate: exact label associations isolate memory storage and overwrite interference.
    # Use the production delta rule; the occupancy wrapper only masks never-written slots.
    @jax.jit
    def write(state, key, value):
        return bank.delta_write_kv_multi(state, key[None, None], value[None, None], jnp.ones((1, 1), bool))[0]

    @jax.jit
    def read(state):
        return bank.read_key(state, jnp.asarray(addresses)[None])[0]

    single = []
    for i in range(len(rows)):
        state = write(blank, jnp.asarray(keys[i]), jnp.asarray(values[i]))
        result = np.asarray(read(state))[groups[i]]
        single.append(float(cosine_matrix(result[None], values[i:i + 1])[0, 0]))
    sequential = []
    for label, order in (("reference_order", range(len(rows))), ("reverse_order", reversed(range(len(rows))))):
        state, latest = blank, {}
        for i in order:
            state = write(state, jnp.asarray(keys[i]), jnp.asarray(values[i]))
            latest[int(groups[i])] = i
        retrieval = np.asarray(read(state))
        for slot, target in sorted(latest.items()):
            candidates = np.flatnonzero(groups == slot)
            scores = cosine_matrix(retrieval[slot:slot + 1], values[candidates])[0]
            best = int(candidates[np.argmax(scores)])
            target_cos = float(cosine_matrix(retrieval[slot:slot + 1], values[target:target + 1])[0, 0])
            alternatives = scores[candidates != target]
            sequential.append({"order": label, "slot": slot, "expected_sentence": target,
                               "nearest_sentence_within_template": best, "correct": best == target,
                               "target_cosine": target_cos, "read_norm": float(np.linalg.norm(retrieval[slot])),
                               "target_margin": float(target_cos - alternatives.max()) if len(alternatives) else None})
    return {"reference_tokens": rows, "template_representatives": list(reps), "template_groups": groups.tolist(),
            "matrices": {k: v.tolist() for k, v in matrices.items()}, "summary": summaries,
            "single_write_recall_cosines": single, "sequential_oracle_writes": sequential,
            "recall_protocol": "All reference sentences once, then reverse order; one real delta commit per sentence, "
                               "including configured per-commit decay. Score latest value per template. No idle ticks, "
                               "no decoder, no confidence gating. This is not episode or task success."}


def measure_tokens(model):
    """Measure the real token writer, including interference within one sentence.

    Context identity follows _v6_context_keys: positions 0 and 1 both use h_0.
    Repeated contexts intentionally overwrite; score the last value at each
    context, not every historical token as though each had a unique address.
    """
    import jax
    import jax.numpy as jnp

    rows = tuple(tuple(row) for row in model.memory_v5_reference_tokens)
    tokens, mask = model.v5_reference_token_rows(model.memory_v5_sentence_len)
    keys, values, valid = model.v6_sentence_token_kv(tokens, mask)
    valid_np = np.asarray(valid)
    locations = np.argwhere(valid_np)
    flat_keys, flat_values = np.asarray(keys)[valid_np], np.asarray(values)[valid_np]
    token_ids = np.asarray(tokens)[valid_np]
    contexts = [tuple(rows[r][:max(1, int(p))]) for r, p in locations]
    representatives = {}
    for i, context in enumerate(contexts):
        representatives.setdefault(context, i)
    reps = list(representatives.values())
    value_reps = [int(np.flatnonzero(token_ids == t)[0]) for t in sorted(set(token_ids.tolist()))]
    bank = model.memory_semantic
    blank = bank.init_state(1)
    hidden = np.asarray(bank.hidden_key(blank, jnp.asarray(flat_keys)[None]))[0]
    matrices = {
        "distinct_context_key": cosine_matrix(flat_keys[reps]),
        "distinct_context_hidden_feature": cosine_matrix(hidden[reps]),
        "distinct_token_value": cosine_matrix(flat_values[value_reps]),
    }
    summaries = {name: pair_summary(mat, np.triu(np.ones_like(mat, bool), 1)) for name, mat in matrices.items()}

    @jax.jit
    def write(state, k, v, keep):
        return bank.delta_write_kv_multi(state, k[None], v[None], keep[None], slot_loop="scan")[0]

    @jax.jit
    def read(state, k):
        return bank.read_key(state, k[None])[0]

    single = []
    for i in range(len(flat_keys)):
        k, v = jnp.asarray(flat_keys[i:i + 1]), jnp.asarray(flat_values[i:i + 1])
        state = write(blank, k, v, jnp.ones((1,), bool))
        single.append(float(cosine_matrix(np.asarray(read(state, k)), flat_values[i:i + 1])[0, 0]))

    # Cosines for ALL original positions after an entire sentence was committed.
    # Unlike isolated association recall, these need not be 1 (overwrites/overlap).
    whole = []
    for r, row in enumerate(rows):
        state = write(blank, keys[r], values[r], valid[r])
        retrieved = np.asarray(read(state, keys[r]))[:len(row)]
        cos = np.diag(cosine_matrix(retrieved, np.asarray(values[r])[:len(row)]))
        whole.append({"sentence": r, "token_cosines": cos.tolist()})

    sequential = []
    for label, order in (("reference_order", range(len(rows))), ("reverse_order", reversed(range(len(rows))))):
        state, latest = blank, {}
        for r in order:
            state = write(state, keys[r], values[r], valid[r])
            for i, (row_id, _) in enumerate(locations):
                if row_id == r:
                    latest[contexts[i]] = i
        targets = list(latest.values())
        retrieved = np.asarray(read(state, jnp.asarray(flat_keys[targets])))
        candidates = flat_values[value_reps]
        candidate_ids = token_ids[value_reps]
        scores = cosine_matrix(retrieved, candidates)
        for j, i in enumerate(targets):
            best = int(candidate_ids[np.argmax(scores[j])])
            expected = int(token_ids[i])
            target_cos = float(cosine_matrix(retrieved[j:j + 1], flat_values[i:i + 1])[0, 0])
            alternatives = scores[j][candidate_ids != expected]
            sequential.append({"order": label, "context_tokens": list(contexts[i]),
                               "expected_token": expected, "nearest_token": best, "correct": best == expected,
                               "target_cosine": target_cos, "read_norm": float(np.linalg.norm(retrieved[j])),
                               "target_margin": float(target_cos - alternatives.max()) if len(alternatives) else None})
    return {"reference_tokens": rows, "valid_token_count": len(locations),
            "context_count": len(representatives), "context_representatives": [list(c) for c in representatives],
            "candidate_token_ids": token_ids[value_reps].tolist(),
            "matrices": {name: mat.tolist() for name, mat in matrices.items()}, "summary": summaries,
            "single_write_recall_cosines": single, "whole_sentence_write_recall": whole,
            "sequential_oracle_writes": sequential,
            "recall_protocol": "Production contextual token keys/values, configured MLP and delta rule. Isolated "
                               "associations, then complete sentences, then all references forward/reversed. "
                               "One decay per sentence commit; no idle ticks. Sequential scoring targets the LAST "
                               "token at each causal context (positions 0 and 1 share h_0 in the existing writer). "
                               "No learned-reader, decoder, or confidence-gate evaluation; not task success."}


def main():
    from openpi.training import config as config_lib
    from openpi.models import tokenizer as tokenizer_lib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_yam_beans0922_ab_snap_mlp3_A")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize", action="store_true")
    source.add_argument("--params", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = config_lib.get_config(args.config_name)
    is_token = config.model.memory_v6_token_writes
    if not config.model.memory_template_read and not is_token:
        raise ValueError("This probe requires direct-template reads or the contextual token writer.")
    print("Loading CPU-only model; no optimizer, data loader, or training.", flush=True)
    model, provenance = load_model(config, args, args.output_dir)
    print("Measuring production token geometry." if is_token else "Measuring production sentence/slot geometry.", flush=True)
    report = measure(model)
    tokenizer = tokenizer_lib.PaligemmaTokenizer()
    sp = next(v for v in vars(tokenizer).values() if hasattr(v, "decode"))
    report["sentences"] = [sp.decode(list(row)).strip() for row in report["reference_tokens"]]
    report.update(schema_version="contextual_token_geometry/1" if is_token else "template_sentence_geometry/1",
                  config_name=args.config_name, provenance=provenance,
                  bank_dims=list(model.memory_semantic.config.dims), alpha_step=model.memory_semantic.config.alpha_step,
                  device="cpu")
    path = args.output_dir / "sentence_geometry.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, summary in report["summary"].items():
        print(name, json.dumps(summary), flush=True)
    recall = report["single_write_recall_cosines"]
    records = report["sequential_oracle_writes"]
    print(f"Isolated-association cosine min/mean: {min(recall):.6f}/{np.mean(recall):.6f}", flush=True)
    print(f"Sequential latest-value identification: {sum(r['correct'] for r in records)}/{len(records)}", flush=True)
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
