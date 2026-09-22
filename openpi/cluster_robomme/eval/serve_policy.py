#!/usr/bin/env python3
"""Serve the v7 MemoryPolicy (scripts/serve_yam_memory.py) for local RoboMME rollouts (cluster_robomme/README.md)."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import traceback

from common import TASKS, decode_policy_request, jsonable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--stage", choices=("A", "B"), default="B")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Finalized numeric STEP directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18767, type=int)
    parser.add_argument("--num-steps", default=10, type=int, help="Action denoising steps")
    parser.add_argument("--diagnostic-oracle-subtask", action="store_true", help="Accept official online labels for an excluded diagnostic only")
    parser.add_argument("--diagnostic-oracle-write", action="store_true",
                        help="Excluded diagnostic (stage A style): the bank is written with the true sentence sent as oracle_write_subtask; the decode stays the model's own")
    parser.add_argument("--show-own-prediction", action="store_true",
                        help="With --diagnostic-oracle-subtask: also decode the model's OWN sentence at every tick (same memory, no write) "
                             "and return it as own_subtask / own_confidence for the record and the video overlay")
    parser.add_argument("--fast-decode", action="store_true",
                        help="Replay diagnostics (09-18): also decode the FAST branch (fast_actions) and report the prompt slot; policy behaviour unchanged")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    root = Path(__file__).resolve().parents[3]
    # memory_project_robomme (2026-09-13): the v7 boba-recipe configs (robomme_config.py); the checkpoint directory
    # names its config, so the served model always uses the exact training recipe.
    # 09-17: the config is the checkpoint's grandparent directory (robomme/checkpoints/<config>/<exp>/<step>); any
    # registered pi05_robomme* config of this task is served (the memory runs are named robomme_MMDD_vN now).
    if not checkpoint.name.isdigit() or checkpoint.parents[2] != root / "robomme/checkpoints":
        parser.error("Checkpoint must be robomme/checkpoints/<config>/<exp>/<step> in the RoboMME copy")
    config = checkpoint.parents[1].name
    if not config.startswith("pi05_robomme"):
        parser.error("Checkpoint config does not look like a RoboMME config")
    if not all((checkpoint / name).is_dir() for name in ("params", "assets", "train_state")):
        parser.error("Checkpoint is incomplete")
    # Match the validated runtime import order. Model implementation is reused unchanged.
    import torch  # noqa: F401
    import jax
    import jax.numpy as jnp
    import numpy as np
    from serve_yam_memory import Args, create_policy
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = create_policy(Args(dir=str(checkpoint), config=config, max_decode_steps=48,
                                num_steps=args.num_steps, warmup=False, fast_decode=args.fast_decode))
    metadata = dict(policy.metadata, task=args.task, checkpoint=str(checkpoint), training_config=config,
                    trained_stage=args.stage, trained_update=int(checkpoint.name),
                    action_space="joint_angle", action_dim=8, prediction_source="decoded_policy_tokens",
                    ground_truth_input=False, max_decode_steps=48, num_steps=args.num_steps,
                    model_backend=jax.default_backend(), diagnostic_oracle_subtask=args.diagnostic_oracle_subtask,
                    diagnostic_oracle_write=args.diagnostic_oracle_write)
    if args.diagnostic_oracle_write:
        metadata.update(write_source="forced_official_online_subtask", benchmark_eligible=False)
    if args.diagnostic_oracle_subtask:
        metadata.update(prediction_source="forced_official_online_subtask", ground_truth_input=True,
                        benchmark_eligible=False, own_prediction_shown=bool(args.show_own_prediction))
        policy._own_prediction = bool(args.show_own_prediction)  # noqa: SLF001
    elif args.show_own_prediction:
        parser.error("--show-own-prediction only makes sense with --diagnostic-oracle-subtask")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            body = json.dumps(jsonable(data), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self.reply(metadata if self.path == "/metadata" else {"ready": True},
                       200 if self.path in ("/metadata", "/healthz") else 404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if self.path == "/reset":
                    result = policy.infer({"reset_memory": True})
                    policy._rng = jax.random.key(int(data["seed"]))
                    self.reply(result)
                elif self.path == "/infer":
                    oracle_write = data.pop("oracle_write_subtask", None) if args.diagnostic_oracle_write else None
                    if args.diagnostic_oracle_write and (not isinstance(oracle_write, str) or not oracle_write.strip() or len(oracle_write) > 2000):
                        raise ValueError("Oracle-write diagnostic requires a nonempty oracle_write_subtask")
                    obs, forced = decode_policy_request(data, args.diagnostic_oracle_subtask)
                    if forced is not None:
                        # Use the unchanged v6 sampler's existing forced-subtask interface.
                        ids = policy._decode_tokenizer.encode(forced.strip() + "\n")
                        length = int(policy._model.causal_token_len)
                        if not 0 < len(ids) <= length:
                            raise ValueError("Official sentence exceeds the model causal buffer")
                        tokens = np.zeros((1, length), np.int32)
                        mask = np.zeros((1, length), bool)
                        tokens[0, :len(ids)], mask[0, :len(ids)] = ids, True
                        policy._forced = (jnp.asarray(tokens), jnp.asarray(mask))
                    result = policy.infer(obs, oracle_write_subtask=oracle_write) if oracle_write is not None else policy.infer(obs)
                    if forced is not None:
                        if result["subtask"] != forced.strip():
                            raise ValueError("Forced sentence does not match sampler output")
                        result.update(subtask_confidence=None, subtask_source="forced_official_online_subtask")
                    self.reply(result)
                else:
                    self.reply({"error": "Unknown endpoint"}, 404)
            except Exception as exc:
                traceback.print_exc()
                self.reply({"error": str(exc)}, 500)

    # One sequential local client owns this episode memory. A separate process per parallel rollout.
    print("READY {} {} http://{}:{}".format(args.task, checkpoint, args.host, args.port), flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
