"""Teacher-forced probe of the button-press rows under TRAINING conditions (09-16, robomme_memory).

Why: every free run since r1/1000 says "pick up ... for the <count> time" through the whole press phase, even with an
oracle bank (press rows exact 0-1 of 5-7), while the training log reports 92-97 % decision exactness with the press rows
being ~25 % of the decision rows. This script feeds the SAME window the training loader would build (label-history
prefill, own-content writes, delay/debounce of the config) through `compute_loss(train=False)` for a window anchored a
few steps before the press onset and prints the per-step teacher-forced exactness of every valid step. If the press rows
are exact here, the free-running miss is a train/eval mismatch (window start, bank history); if they are wrong here too,
the CE on the press rows is not being learned.

  .venv/bin/python cluster_robomme/probe_press_tf.py --config-name <cfg> --params <ckpt>/params --episodes 0,1,3,7 \
      --anchor-steps 2,10 --manifest robomme/metadata/PickXtimes/manifest.json --sidecar robomme/metadata/PickXtimes/subtasks_official.json
"""
import argparse
import json
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

import openpi.models.model as _model
import openpi.shared.project_paths as project_paths
import openpi.training.config as _config
import openpi.training.data_loader as data_loader_lib

STOP = (108, 1, 0)


def _text(sp, ids):
    ids = [int(t) for t in ids]
    while ids and ids[-1] in STOP:
        ids.pop()
    return sp.decode(ids).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--params", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", default="0,1,3,7")
    ap.add_argument("--anchor-steps", default="2,10", help="window starts this many memory steps BEFORE the press onset")
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--sidecar", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="json summary path")
    args = ap.parse_args()
    t0 = time.time()
    cfg = _config.get_config(args.config_name)
    params = _model.restore_params(args.params, restore_type=np.ndarray)
    model = cfg.model.load(params)
    model.eval()
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader_lib.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    tds = data_loader_lib.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, *data_config.model_transforms.inputs]
    )
    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(project_paths.project_path("v35/cache/openpi/big_vision/paligemma_tokenizer.model"))
    )
    manifest = json.loads(args.manifest.read_text())
    sidecar = json.loads(args.sidecar.read_text())
    episodes = sorted([e for e in manifest["episodes"] if e.get("include", True)], key=lambda e: e["episode_index"])
    stride = int(data_config.memory_stride_frames)
    sent_len = int(cfg.model.memory_v5_sentence_len)
    print(f"setup {time.time() - t0:.0f}s: stride {stride}, delay {getattr(cfg.model, 'memory_v5_write_delay_steps', 0)}, "
          f"debounce {getattr(cfg.model, 'memory_v7_write_debounce_steps', 1)}, own content "
          f"{not getattr(cfg.model, 'memory_v5_own_commit_label_content', False)}", flush=True)

    @nnx.jit
    def loss_fn(model, obs, actions):
        return model.compute_loss(jax.random.key(0), obs, actions, train=False)

    results = []
    for ep_idx in [int(x) for x in args.episodes.split(",")]:
        episode = episodes[ep_idx]
        assert episode["episode_index"] == ep_idx
        start = sum(int(e["expected_num_frames"]) for e in episodes[:ep_idx])
        length = int(episode["expected_num_frames"])
        segments = sidecar["episodes"][episode["stable_id"]]["segments"]
        press = next(s for s in segments if s["sentence"].startswith("press"))
        press_step = -(-int(press["start"]) // stride)  # first memory step inside the press segment
        for a in [int(x) for x in args.anchor_steps.split(",")]:
            frame = max(0, (press_step - a) * stride)
            item = tds[start + frame]
            batched = jax.tree.map(lambda x: np.asarray(x)[None], item)
            obs = _model.Observation.from_dict(batched)
            actions = jnp.asarray(batched["actions"])
            t1 = time.time()
            out = loss_fn(model, obs, actions)
            out = {k: np.asarray(v) for k, v in out.items() if k in ("v5_exact_decision_steps", "v4_decision_active_steps", "v5_step_ce_steps", "v5_exact_evidence_steps")}
            exact = out["v5_exact_decision_steps"][:, 0]
            active = out["v4_decision_active_steps"][:, 0]
            ce = out["v5_step_ce_steps"][:, 0]
            ev_exact = out["v5_exact_evidence_steps"][:, 0]
            step_mask = np.asarray(obs.seq_step_mask)[0]
            causal = np.asarray(obs.tokenized_causal)[0]
            span = (np.asarray(obs.tokenized_causal_mask)[0] & ~np.asarray(obs.causal_fast_mask)[0])[:, :sent_len]
            prefill = []
            if obs.memory_v5_prefill_tokens is not None:
                pt = np.asarray(obs.memory_v5_prefill_tokens)[0]
                pm = np.asarray(obs.memory_v5_prefill_mask)[0]
                prefill = [_text(sp, pt[i][pm[i]]) for i in range(pt.shape[0]) if pm[i].any()]
            pending = ""
            if obs.memory_v5_pending_tokens is not None:
                pd, pdm = np.asarray(obs.memory_v5_pending_tokens)[0], np.asarray(obs.memory_v5_pending_mask)[0]
                pending = _text(sp, pd[pdm]) if pdm.any() else ""
            print(f"\n== ep {ep_idx} anchor -{a} steps: window frame {frame} (press onset step {press_step} = frame {press_step * stride}), "
                  f"{int(step_mask.sum())} valid steps, loss {time.time() - t1:.0f}s\n   prefill[{len(prefill)}]: {' | '.join(s[:38] for s in prefill[-4:])}  pending: {pending!r}", flush=True)
            rows = []
            for t in range(len(step_mask)):
                if not step_mask[t]:
                    continue
                label = _text(sp, causal[t][:sent_len][span[t]])
                kind = label.split(" ")[0] if label else "?"
                ok = bool(exact[t]) if active[t] else bool(ev_exact[t])
                rows.append({"step": t, "frame": frame + t * stride, "label": label, "decision": bool(active[t]), "exact": ok, "ce": float(ce[t])})
                print(f"   t{t:2d} f{frame + t * stride:4d} {'D' if active[t] else ' '} {'OK' if ok else '--'} ce {ce[t]:5.2f}  {label}", flush=True)
            press_rows = [r for r in rows if r["label"].startswith("press")]
            other_dec = [r for r in rows if r["decision"] and not r["label"].startswith("press")]
            summ = {"episode": ep_idx, "anchor": a, "frame": frame, "press_exact": sum(r["exact"] for r in press_rows), "press_rows": len(press_rows),
                    "other_decision_exact": sum(r["exact"] for r in other_dec), "other_decision_rows": len(other_dec), "prefill": prefill, "pending": pending, "rows": rows}
            results.append(summ)
            print(f"   >> press rows exact {summ['press_exact']}/{summ['press_rows']}, other decision rows {summ['other_decision_exact']}/{summ['other_decision_rows']}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"config": args.config_name, "params": str(args.params), "results": results}, indent=1))
    print("\nSUMMARY press rows exact under training conditions:")
    for r in results:
        print(f"  ep {r['episode']} anchor -{r['anchor']:2d}: press {r['press_exact']}/{r['press_rows']}  other decision {r['other_decision_exact']}/{r['other_decision_rows']}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
