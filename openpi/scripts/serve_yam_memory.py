"""Serve a pi05_yam_mem_* memory checkpoint over websocket, threading the Titans memory.

Like scripts/serve_yam_subtask.py, but the policy runs `Pi0.sample_with_memory`: every request
reads the memory, decodes the subtask, denoises the actions and then writes the frame's hidden
representation into the per-episode memory state, which is threaded across requests. v3 writes
the raw layer-8 top-camera states; v3.1 writes the memory-token block's final-normalized output.
Each response
carries "subtask", "surprise" (1-ish = novel, ~0 = recalled), the write gates and the running
write count. A request containing "reset_memory": true re-initializes the memory (send one at
every episode start); a bare {"reset_memory": true} request (no images) just resets and returns.
For RTC-trained checkpoints, an optional "action_prefix" carries the still-executing portion
of the previous chunk plus the anticipated inference delay.

The client controls the write cadence: one infer call = one memory write, so call at the
training stride (memory_stride_frames=10 @ 30 Hz -> ~0.33 s between calls).

Usage (on the GPU box):
    uv run scripts/serve_yam_memory.py \
        --dir checkpoints/pi05_yam_mem_v31/<exp>/<step> \
        --config pi05_yam_mem_v31
"""

from collections.abc import Mapping
import dataclasses
import logging
import socket
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.rtc as _rtc
import openpi.models.tokenizer as _tokenizer
import openpi.policies.policy as _policy
from openpi.serving import websocket_policy_server
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class Args:
    # Checkpoint directory, e.g. checkpoints/pi05_yam_mem_v3/<exp>/<step> or
    # checkpoints/pi05_yam_mem_v31/<exp>/<step>.
    dir: str
    # Keep v3 as the default until an explicitly selected v3.1 checkpoint exists.
    config: str = "pi05_yam_mem_v3"
    port: int = 8000
    max_decode_steps: int = 10
    # Flow-matching denoising steps of the action expert (training-time default 10). 2026-09-06: the B9 server
    # measured 240 ms per request on an H200 (the RTC client tolerates ~200 ms); 6 steps trims ~40 ms.
    num_steps: int = 10
    force_subtask: str = ""
    """Diagnostic: skip the sentence decode and condition the action expert on this FIXED sentence at every step
    (e.g. "yellow go: pick up the scoop, scoop 2 times" to exercise the pick-up skill; "scoop 1 of 2: dig and carry"
    for the dig). Must be one of the trained sentences. 2026-09-06 22:12, user: with --zero-read alone the
    decoder said "done" and the arm never moved."""
    zero_read: bool = False
    """Diagnostic: zero the semantic-bank READ content before the layer-8 injection (writes and the decoded
    sentence still happen, the count will not work). Isolates whether the memory reads degrade the low-level
    skills (2026-09-06 21:55, user: pick-up/dig look worse than the plain pi05 baseline)."""
    # Run synthetic requests before serving so the JIT compile (minutes) happens here, not on the
    # robot's first request; the memory is reset afterwards (ported from v4).
    warmup: bool = True
    write_conf: float | None = None
    """Override the checkpoint's memory_v5_write_conf (0.9 in the v6 task1 configs): a sentence enters the bank only
    when its mean token probability reaches this. 2026-09-10 robot test: notes rarely appeared / were not committed."""
    log_steps: bool = True
    """Log every request's decoded sentence, confidence, commit flag and bank size (diagnosis on the robot)."""
    fast_decode: bool = False
    """Replay diagnostics (09-18): also decode the trained FAST branch after the sentence and return it as
    `fast_actions` (unnormalized, [action_horizon, dim]) next to the action expert's `actions`; the response also
    carries `prompt_prev_subtask` (the "Last:" slot content). The sentence, the notes and the expert actions are
    unchanged; each request takes longer (up to causal_token_len decode steps)."""


def _build_server_metadata(train_config: Any, data_config: Any, *, simulated_delay: int | None) -> dict[str, Any]:
    """Publish the training semantics needed to reject mismatched clients/checkpoints."""
    metadata: dict[str, Any] = dict(train_config.policy_metadata or {})
    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    metadata.update(
        {
            "config_name": train_config.name,
            "memory_architecture": memory_architecture,
            "memory_write_source": str(getattr(train_config.model, "memory_write_source", "raw_hidden")),
            "memory_query_tokens": (
                int(getattr(train_config.model, "memory_query_tokens", 0))
                if memory_architecture == "v32_layer8_dual_query"
                else None
            ),
            "action_horizon": int(train_config.model.action_horizon),
            "rtc_enabled": simulated_delay is not None,
            "rtc_max_delay": simulated_delay,
            "rtc_delay_semantics": "inclusive_max",
            "memory_stride_frames": int(data_config.memory_stride_frames),
            "memory_v5_sentence_bank": bool(getattr(train_config.model, "memory_v5_sentence_bank", False)),
            # 0920_v0 short visual history: the client keeps the ring buffer and sends observation/history_<i> (oldest
            # first, `image_history_stride` frames apart, the newest one stride before the current frame) + history_valid.
            "image_history_frames": int(getattr(data_config, "memory_image_history_frames", 0)),
            "image_history_stride": int(getattr(data_config, "memory_image_history_stride", 0)),
        }
    )
    return metadata


class V5SentenceMemory:
    """Per-episode carry of the v5 SENTENCE bank at inference (cluster_v5/README.md §5).

    Mirrors the training scan exactly: the sentence WRITTEN at a step is the one decoded at the
    previous step (one-step delay, A4), it is committed only when it differs from the sentence
    before it and was decoded with mean token probability >= write_conf; every other valid step
    is one analytic decay of the bank. The pending (last decoded) sentence is also what the A6
    read queries condition on. `write_fn(tokens[1, L], mask[1, L], state, commit[1]) -> (state,
    applied[1])` must apply the commit or the decay, like `Pi0.v5_semantic_write`."""

    def __init__(self, *, init_state, write_fn, sentence_len: int, write_conf: float, delay_steps: int, decode_text, prev_is_committed: bool = False,
                 debounce_steps: int = 1, vocab_rows=(), retract_steps: int = 0, retract_fn=None, delta_fn=None,
                 first_write_debounce_steps: int | None = None, write_every_step: bool = False):
        """v7 robomme (09-17) generic write rules, the same as the training scan and scripts/v5_heldout_video.py:
        `debounce_steps` = commit only a sentence decoded N steps in a row; `vocab_rows` = if non-empty, commit only a
        sentence that equals one of these token tuples; `retract_steps` = K: an A -> B -> A flip-back within K steps
        of B's commit ERASES B (`retract_fn(state, last_delta, age) -> state`, `delta_fn(before, after) -> delta`)."""
        self._init_state = init_state
        self._write = write_fn
        self._len = int(sentence_len)
        self._conf = float(write_conf)
        self._delay = int(delay_steps)
        self._prev_is_committed = bool(prev_is_committed)
        self._decode_text = decode_text
        self._debounce = max(1, int(debounce_steps))
        # first-note confirmation (09-17): while the bank is empty, require this many identical consecutive decodes
        self._first_debounce = max(1, int(first_write_debounce_steps)) if first_write_debounce_steps is not None else self._debounce
        self._vocab = {tuple(int(t) for t in row) for row in vocab_rows}
        # 0920_v0 (memory_v7_write_every_step, fixed 09-21): every tick with a decoded sentence is a write, changed or
        # not, exactly like the training scan (`write_trigger = sentence_changed | has_span`); the confidence gate still
        # applies (0.0 in that recipe). Only a real change is appended to the readable `bank` list.
        self._every_step = bool(write_every_step)
        self._retract = int(retract_steps)
        self._retract_fn = retract_fn
        self._delta_fn = delta_fn
        self.reset()

    def reset(self) -> None:
        self.state = self._init_state()
        self.prev_tokens = np.zeros((1, self._len), dtype=np.int32)
        self.pending_tokens = np.zeros((1, self._len), dtype=np.int32)
        self.pending_mask = np.zeros((1, self._len), dtype=bool)
        self.pending_conf = False
        self.bank: list[str] = []
        self.commits = 0
        self.steps = 0
        self.last_cand = None  # v7 debounce: previous step's candidate tokens
        self.cand_streak = 0
        self.committed_tok: list[np.ndarray] = []  # token rows of the notes in `bank`
        self.last_delta = None  # v7 retraction: delta of the newest note, and its age in steps
        self.commit_age = 10**6
        self.retractions = 0
        self.vocab_rejections = 0

    @property
    def query_prev(self) -> tuple[np.ndarray, np.ndarray]:
        """What the A6 read queries condition on, as in the training scan: the pending (one-step-delayed)
        sentence with delay 1, otherwise the previous sentence (last produced, or last committed under
        the retry rule). Fixed 2026-09-05 12:40: delay 0 used to pass the never-filled pending slot."""
        if self._delay == 1:
            return self.pending_tokens, self.pending_mask
        return np.maximum(self.prev_tokens, 0), self.prev_tokens > 0

    def step(self, tokens: np.ndarray, mask: np.ndarray, probs: np.ndarray) -> dict:
        """Apply one memory step with this step's decoded span (1-D arrays over the causal buffer)."""
        n = min(self._len, int(np.asarray(tokens).shape[0]))
        produced_tokens = np.zeros((1, self._len), dtype=np.int32)
        produced_mask = np.zeros((1, self._len), dtype=bool)
        produced_mask[0, :n] = np.asarray(mask, dtype=bool)[:n]
        produced_tokens[0, :n] = np.where(produced_mask[0, :n], np.asarray(tokens)[:n], 0)
        confidence = float(np.mean(np.asarray(probs)[np.asarray(mask, dtype=bool)])) if np.any(mask) else 0.0
        produced_conf = confidence >= self._conf
        if self._delay == 1:
            cur_tokens, cur_mask, cur_conf = self.pending_tokens, self.pending_mask, self.pending_conf
        else:
            cur_tokens, cur_mask, cur_conf = produced_tokens, produced_mask, produced_conf
        has_span = bool(np.any(cur_mask))
        changed = has_span and bool(np.any(cur_tokens != self.prev_tokens))
        commit = (changed or (self._every_step and has_span)) and bool(cur_conf)
        # v7 debounce: the candidate must have been decoded `debounce` steps in a row
        same = self.last_cand is not None and has_span and bool(np.array_equal(cur_tokens, self.last_cand))
        self.cand_streak = self.cand_streak + 1 if same else (1 if has_span else 0)
        self.last_cand = cur_tokens.copy()
        debounced = False
        if commit and self.cand_streak < (self._first_debounce if not self.committed_tok else self._debounce):
            commit = False
            debounced = True  # keep `prev` as it is: the same candidate must still count as a change next step
        # v7 vocabulary gate: only complete reference sentences enter the bank
        if commit and self._vocab and tuple(int(t) for t in cur_tokens[0][cur_mask[0]]) not in self._vocab:
            commit = False
            self.vocab_rejections += 1
        # v7 flip-back retraction: A -> B -> A within K steps erases B instead of writing A again
        retracted = False
        if (commit and self._retract > 0 and self.last_delta is not None and len(self.committed_tok) >= 2
                and self.commit_age + 1 <= self._retract and bool(np.array_equal(cur_tokens, self.committed_tok[-2]))):
            commit = False
            retracted = True
        before = self.state
        self.state, applied = self._write(cur_tokens, cur_mask, self.state, np.asarray([commit]))
        applied = bool(np.asarray(applied).reshape(-1)[0])
        if retracted:
            self.state = self._retract_fn(self.state, self.last_delta, self.commit_age + 1)
            self.bank.pop(); self.committed_tok.pop()
            self.last_delta = None; self.commit_age = 10**6; self.retractions += 1
        elif applied:
            if self._delta_fn is not None:
                self.last_delta = self._delta_fn(before, self.state)
            self.commit_age = 0
            self.committed_tok.append(cur_tokens.copy())
        else:
            self.commit_age = min(self.commit_age + 1, 10**6)
        if applied or (not self._prev_is_committed and not debounced):
            self.prev_tokens = cur_tokens
        if retracted:
            self.prev_tokens = self.committed_tok[-1].copy()
        if self._delay == 1:
            self.pending_tokens, self.pending_mask, self.pending_conf = produced_tokens, produced_mask, produced_conf
        if applied:
            text = self._decode_text(cur_tokens[0][cur_mask[0]].tolist())
            if changed or not self.bank or self.bank[-1] != text:  # write-every-step rewrites are not new notes
                self.bank.append(text)
            self.commits += 1
        self.steps += 1
        return {
            "sentence": self._decode_text(produced_tokens[0][produced_mask[0]].tolist()),
            "confidence": confidence,
            "changed": changed,
            "committed": applied,
            "retracted": retracted,
            "bank": list(self.bank),
            "writes": self.commits,
            "retractions": self.retractions,
            "vocab_rejections": self.vocab_rejections,
        }


class MemoryPolicy(_policy.Policy):
    """Policy that threads the Titans memory state across requests.

    Responses carry the decoded subtask, the pre-write surprise and the (frozen) write gates.
    """

    def __init__(
        self,
        model,
        *,
        decode_tokenizer,
        stop_token: int,
        max_decode_steps: int,
        num_steps: int = 10,
        zero_read: bool = False,
        force_subtask: str = "",
        write_conf: float | None = None,
        log_steps: bool = False,
        action_horizon: int,
        action_dim: int,
        raw_action_dim: int,
        simulated_delay: int | None,
        fast_decode: bool = False,
        fast_tokenizer=None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self._decode_tokenizer = decode_tokenizer
        # fast_decode (09-18): decode the FAST branch too; "|" (the last id before eos in the trained postfix) ends it
        self._fast_decode = bool(fast_decode)
        self._fast_tokenizer = fast_tokenizer
        self._fast_stop_token = int(decode_tokenizer.encode("|")[-1])
        self._causal_len = int(getattr(model, "causal_token_len", 0) or 0)
        if self._fast_decode and (fast_tokenizer is None or not self._causal_len):
            raise ValueError("--fast-decode needs the FAST tokenizer and a memory model with a causal buffer")
        # v7 prompt slot (09-18): models with prompt_slot_len > 0 get "Last: <newest committed note | none>" per tick
        self._prompt_slot = int(getattr(model, "prompt_slot_len", 0) or 0) > 0
        self._stop_token = stop_token
        self._write_conf_override = write_conf
        self._log_steps = bool(log_steps)
        self._step_counter = 0
        self._max_decode_steps = max_decode_steps
        self._num_steps = int(num_steps)
        self._zero_read = bool(zero_read)
        self._forced = None
        # 09-19 20:40 (user: "add a mode using correct notes as current subtask, but on the video also show the
        # model's own prediction"): when a sentence is forced and this flag is set, the model is ALSO decoded freely
        # on the same observation and memory state (result discarded: no write, no state change) and its sentence is
        # returned as own_subtask / own_confidence for the record and the video overlay.
        self._own_prediction = False
        if force_subtask.strip():
            ids = decode_tokenizer.encode(force_subtask.strip() + "\n")
            length = int(model.causal_token_len)
            if not 0 < len(ids) <= length:
                raise ValueError(f"--force-subtask {force_subtask!r}: {len(ids)} tokens, limit {length}")
            tokens = np.zeros((1, length), dtype=np.int32); tokens[0, : len(ids)] = ids
            mask = np.zeros((1, length), dtype=bool); mask[0, : len(ids)] = True
            self._forced = (jnp.asarray(tokens), jnp.asarray(mask))
            logging.info("forced subtask at every step: %r (%d tokens)", force_subtask.strip(), len(ids))
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._raw_action_dim = raw_action_dim
        self._simulated_delay = simulated_delay
        self._sample = nnx_utils.module_jit(
            model.sample_with_memory,
            static_argnames=("stop_token", "max_decode_steps", "write_mode", "zero_read", "fast_decode", "fast_stop_token"),
        )
        self._init_state = lambda: model.memory.init_state(1)
        self._lock = threading.Lock()
        self._memory_state = self._init_state()
        self._writes = 0
        # v5 sentence bank (cluster_v5/README.md): the semantic bank is read by sample_with_memory
        # and written here with the training write rule; the visual bank is frozen (its
        # injection is off in every v5 stage >= A4).
        self._v5: V5SentenceMemory | None = None
        if getattr(model, "memory_v5_sentence_bank", False):

            @nnx.jit
            def _v5_write(model, tokens, mask, state, commit):
                # v5 pooled write (A8-aware) or, with memory_v6_token_writes, the v6 token-level write.
                return model.v5_commit_sentence(state, tokens, mask, commit)

            def write_fn(tokens, mask, state, commit):
                new_state, applied = _v5_write(
                    model, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(mask), state, jnp.asarray(commit)
                )
                jax.block_until_ready(new_state)
                return new_state, np.asarray(applied)

            # v7 robomme write rules (09-17): debounce / vocabulary gate / flip-back retraction from the config
            sem_out = model.memory_semantic._output_weight_name  # noqa: SLF001
            rho_one = float(1.0 - model.memory_semantic.config.alpha_step)

            def delta_fn(before, after):
                return np.asarray(after.fast_weights[sem_out], dtype=np.float32) - rho_one * np.asarray(before.fast_weights[sem_out], dtype=np.float32)

            def retract_fn(state, last_delta, age):
                w3 = np.asarray(state.fast_weights[sem_out], dtype=np.float32) - (rho_one ** int(age)) * last_delta
                return model.memory_semantic._canonical_delta_state(state, jnp.asarray(w3))  # noqa: SLF001

            self._v5 = V5SentenceMemory(
                init_state=lambda: model.memory_semantic.init_state(1),
                write_fn=write_fn,
                sentence_len=int(model.memory_v5_sentence_len),
                write_conf=float(model.memory_v5_write_conf if write_conf is None else write_conf),
                prev_is_committed=bool(getattr(model, "memory_v5_prev_is_committed", False)),
                delay_steps=int(getattr(model, "memory_v5_write_delay_steps", 0)),
                decode_text=lambda ids: decode_tokenizer.decode(ids).strip(),
                debounce_steps=int(getattr(model, "memory_v7_write_debounce_steps", 1)),
                first_write_debounce_steps=getattr(model, "memory_v7_first_write_debounce_steps", None),
                write_every_step=bool(getattr(model, "memory_v7_write_every_step", False)),
                vocab_rows=tuple(getattr(model, "memory_v5_reference_tokens", ())) if getattr(model, "memory_v7_write_vocab_only", False) else (),
                retract_steps=int(getattr(model, "memory_v7_write_retract_steps", 0)),
                retract_fn=retract_fn, delta_fn=delta_fn,
            )
            logging.info("v7 write rules: debounce=%d vocab_only=%s retract=%d", self._v5._debounce, bool(self._v5._vocab), self._v5._retract)  # noqa: SLF001

    @staticmethod
    def _integer_scalar(value: Any, *, name: str) -> int:
        """Return a protocol integer without silently truncating floats or accepting booleans."""
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "iu":
            raise ValueError(f"action_prefix.{name} must be an integer scalar, got {value!r}")
        return int(array)

    def _prepare_action_prefix(self, inputs: dict, prefix: Any) -> _rtc.ActionPrefix:
        if not isinstance(prefix, Mapping):
            raise ValueError(f"action_prefix must be a mapping, got {type(prefix).__name__}")

        missing = {"actions", "delay", "prefix_length"} - prefix.keys()
        if missing:
            raise ValueError(f"action_prefix is missing required fields: {sorted(missing)}")

        delay = self._integer_scalar(prefix["delay"], name="delay")
        prefix_length = self._integer_scalar(prefix["prefix_length"], name="prefix_length")
        if not 0 <= delay <= prefix_length <= self._action_horizon:
            raise ValueError(
                "action_prefix must satisfy 0 <= delay <= prefix_length <= action_horizon; "
                f"got delay={delay}, prefix_length={prefix_length}, "
                f"action_horizon={self._action_horizon}"
            )
        if self._simulated_delay is None:
            raise ValueError("action_prefix requires a policy configured with simulated_delay")
        if delay > self._simulated_delay:
            raise ValueError(
                f"action_prefix.delay ({delay}) exceeds the configured inclusive RTC maximum ({self._simulated_delay})"
            )

        raw_actions = np.asarray(prefix["actions"])
        expected_raw_shape = (self._action_horizon, self._raw_action_dim)
        if raw_actions.shape != expected_raw_shape:
            raise ValueError(f"action_prefix.actions must have shape {expected_raw_shape}, got {raw_actions.shape}")
        if raw_actions.dtype.kind not in "fiu" or not np.all(np.isfinite(raw_actions)):
            raise ValueError("action_prefix.actions must contain only finite numeric values")

        # Use the exact observation/action input pipeline used for training. In particular,
        # the client sends absolute actions in robot units; DeltaActions, Normalize and
        # PadStatesAndActions must run before the prefix reaches the model.
        rtc_inputs = jax.tree.map(lambda x: x, inputs)
        rtc_inputs["actions"] = np.asarray(raw_actions, dtype=np.float32).copy()
        rtc_inputs = self._input_transform(rtc_inputs)
        transformed_actions = np.asarray(rtc_inputs["actions"])
        expected_model_shape = (self._action_horizon, self._action_dim)
        if transformed_actions.shape != expected_model_shape:
            raise ValueError(
                "transformed action_prefix.actions has an unexpected shape: "
                f"expected {expected_model_shape}, got {transformed_actions.shape}"
            )
        if not np.all(np.isfinite(transformed_actions)):
            raise ValueError("transformed action_prefix.actions contains non-finite values")

        action_prefix = _rtc.ActionPrefix(
            actions=transformed_actions,
            delay=np.asarray(delay, dtype=np.int32),
            prefix_length=np.asarray(prefix_length, dtype=np.int32),
        )
        _rtc.validate_action_prefix(
            action_prefix,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
        )
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], action_prefix)

    @override
    def _newest_note_text(self) -> str:
        """Text of the newest committed note in the sentence bank ("none" when the bank is empty)."""
        v5 = getattr(self, "_v5", None)
        if v5 is None or not v5.committed_tok:
            return "none"
        row = np.asarray(v5.committed_tok[-1]).reshape(-1)  # rows are stored as [1, sentence_len]
        ids = [int(t) for t in row if int(t) > 0]
        text = self._decode_tokenizer.decode(ids).strip() if ids else ""
        return text or "none"

    def infer(self, obs: dict, *, noise: np.ndarray | None = None, oracle_write_subtask: str | None = None) -> dict:  # type: ignore[misc]
        """`oracle_write_subtask` (09-17 diagnostic, stage-A style): the model decodes freely (that sentence is returned and
        scored by the caller) but the TRUE sentence goes through the write rule into the bank, as label writes do in
        stage A training; the read queries then condition on the label as in the training scan."""
        inputs = jax.tree.map(lambda x: x, obs)  # copy: transforms may modify in place
        if inputs.pop("reset_memory", False):
            self._step_counter = 0
            with self._lock:
                self._memory_state = self._init_state()
                self._writes = 0
                if self._v5 is not None:
                    self._v5.reset()
            logging.info("memory reset")
            if "observation/image" not in inputs:  # bare reset ping
                return {"reset": True, "writes": 0}

        prefix = inputs.pop("action_prefix", None)
        action_prefix = self._prepare_action_prefix(inputs, prefix) if prefix is not None else None
        prev_text = None
        if self._prompt_slot:
            # v7 prompt slot (09-18): the context's "Last:" slot carries the newest note in the bank (or "none"),
            # exactly as in training (previous step's label in stage A, own committed note in stage B).
            prev_text = self._newest_note_text()
            inputs["prev_subtask"] = prev_text
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        with self._lock:
            self._rng, sample_rng = jax.random.split(self._rng)
            v5_kwargs = {}
            if self._v5 is not None:
                prev_tokens, prev_mask = self._v5.query_prev
                v5_kwargs = {
                    "semantic_state": self._v5.state,
                    "v5_prev_tokens": jnp.asarray(prev_tokens, dtype=jnp.int32),
                    "v5_prev_mask": jnp.asarray(prev_mask),
                    "write_mode": "frozen",
                }
            own_subtask, own_confidence = None, None
            if self._forced is not None and self._own_prediction:
                # what the model would say on its own, from the same memory state (no write, state discarded)
                self._rng, own_rng = jax.random.split(self._rng)
                _, _, own_aux = self._sample(
                    own_rng, observation, self._memory_state, stop_token=self._stop_token,
                    max_decode_steps=self._max_decode_steps, num_steps=self._num_steps, zero_read=self._zero_read,
                    fast_decode=False, fast_stop_token=self._fast_stop_token, forced_subtask_tokens=None,
                    forced_subtask_mask=None, action_prefix=action_prefix, **v5_kwargs)
                o_tokens = np.asarray(own_aux["tokens"])[0]; o_mask = np.asarray(own_aux["token_mask"])[0]
                own_subtask = self._decode_tokenizer.decode(o_tokens[o_mask].tolist()).strip()
                if "token_prob" in own_aux:
                    o_prob = np.asarray(own_aux["token_prob"])[0][o_mask]
                    own_confidence = float(o_prob.min()) if o_prob.size else None  # the least certain token of the sentence
            actions, new_state, aux = self._sample(
                sample_rng,
                observation,
                self._memory_state,
                stop_token=self._stop_token,
                max_decode_steps=self._causal_len if self._fast_decode else self._max_decode_steps,
                num_steps=self._num_steps,
                zero_read=self._zero_read,
                fast_decode=self._fast_decode,
                fast_stop_token=self._fast_stop_token,
                forced_subtask_tokens=None if self._forced is None else self._forced[0],
                forced_subtask_mask=None if self._forced is None else self._forced[1],
                action_prefix=action_prefix,
                **v5_kwargs,
            )
            jax.block_until_ready(new_state)
            self._memory_state = new_state
            tokens = np.asarray(aux["tokens"])[0]
            mask = np.asarray(aux["token_mask"])[0]
            v5_info = None
            if self._v5 is not None:
                # a forced sentence carries no per-token probabilities (teacher-forced): treat it as confident
                probs = np.asarray(aux["token_prob"])[0] if "token_prob" in aux else np.ones(mask.shape, dtype=np.float32)
                if oracle_write_subtask is not None:
                    o_ids = self._decode_tokenizer.encode(oracle_write_subtask.strip() + "\n")
                    o_tokens = np.zeros(mask.shape, dtype=np.int32); o_mask = np.zeros(mask.shape, dtype=bool)
                    if not 0 < len(o_ids) <= o_tokens.shape[0]:
                        raise ValueError("Oracle-write sentence exceeds the model causal buffer")
                    o_tokens[:len(o_ids)], o_mask[:len(o_ids)] = o_ids, True
                    v5_info = self._v5.step(o_tokens, o_mask, np.ones(mask.shape, dtype=np.float32))
                    v5_info["oracle_write"] = oracle_write_subtask.strip()
                else:
                    v5_info = self._v5.step(tokens, mask, probs)
                writes = v5_info["writes"]
            else:
                self._writes += 1
                writes = self._writes
        model_time = time.monotonic() - start_time

        subtask = self._decode_tokenizer.decode(tokens[mask].tolist()).strip()

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask
        outputs["writes"] = writes
        if own_subtask is not None:
            outputs["own_subtask"] = own_subtask
            outputs["own_confidence"] = own_confidence
        if prev_text is not None:
            outputs["prompt_prev_subtask"] = prev_text
        if self._fast_decode and "fast_tokens" in aux:
            # the FAST branch: "Action: " + action tokens + "|"; decoded with the training tokenizer, then through the
            # same output transform (unnormalize) as the expert actions
            f_tokens = np.asarray(aux["fast_tokens"])[0]
            f_mask = np.asarray(aux["fast_mask"])[0]
            f_text = self._decode_tokenizer.decode(f_tokens[f_mask].tolist())
            fast_ok = "Action: " in f_text and "|" in f_text
            # the FAST tokens were made from the raw (8-dim) normalized actions, before the pad to the model width
            fast_norm = self._fast_tokenizer.extract_actions(f_tokens[f_mask], self._action_horizon, self._raw_action_dim)
            fast_ok = fast_ok and bool(np.any(np.asarray(fast_norm) != 0))  # the FAST decoder returns zeros on a malformed token run
            fast_out = self._output_transform({"state": np.asarray(inputs["state"][0]), "actions": np.asarray(fast_norm, dtype=np.float32)})
            outputs["fast_actions"] = np.asarray(fast_out["actions"], dtype=np.float32)
            outputs["fast_ok"] = bool(fast_ok)
            outputs["fast_tokens"] = int(f_mask.sum())
        self._step_counter += 1
        if self._log_steps:
            if v5_info is not None:
                logging.info("step %d: %r conf=%.2f changed=%s committed=%s writes=%d bank=%d (%.0f ms)", self._step_counter, subtask,
                             float(v5_info["confidence"]), bool(v5_info["changed"]), bool(v5_info["committed"]), int(writes),
                             len(v5_info.get("bank", [])), 1000 * model_time)
            else:
                logging.info("step %d: %r writes=%d (%.0f ms)", self._step_counter, subtask, int(writes), 1000 * model_time)
        if v5_info is not None:
            outputs["subtask_confidence"] = v5_info["confidence"]
            outputs["bank"] = v5_info["bank"]
            outputs["memory"] = {"changed": v5_info["changed"], "committed": v5_info["committed"]}
            if "oracle_write" in v5_info:
                outputs["oracle_write"] = v5_info["oracle_write"]
            # legacy client fields (v3 clients read them): no surprise/gates in the sentence bank
            outputs["surprise"] = 0.0
            outputs["gates"] = {}
        else:
            outputs["surprise"] = float(aux["surprise"][0])
            outputs["gates"] = {k: float(np.asarray(aux[k]).mean()) for k in ("theta", "eta", "alpha")}
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


def create_policy(args: Args) -> MemoryPolicy:
    train_config = _config.get_config(args.config)
    assert train_config.model.predict_with_memory, f"config {args.config} was not built with predict_with_memory"
    checkpoint_dir = download.maybe_download(args.dir)

    logging.info("Loading model (float32: the memory's inner GD is validated in f32)...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float32))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    memory_write_source = str(getattr(train_config.model, "memory_write_source", "raw_hidden"))
    gate = np.asarray(model.memory_gate.value) if hasattr(model, "memory_gate") else np.zeros((1,))
    logging.info(
        "config=%s | memory_architecture=%s | memory_write_source=%s | memory_layer=%d | "
        "memory_gate norm %.4f max|g| %.5f (0 = memory content unused)",
        train_config.name,
        memory_architecture,
        memory_write_source,
        model.memory_layer,
        np.linalg.norm(gate),
        np.abs(gate).max(),
    )
    max_decode_steps = args.max_decode_steps
    if getattr(train_config.model, "memory_v5_sentence_bank", False):
        # v5 sentences ("inspect both bins: banana left, grey pepper box right") need ~12 tokens;
        # the training scan decodes up to 24.
        max_decode_steps = max(max_decode_steps, 24)
        logging.info(
            "v5 sentence bank: delay=%d write_conf=%.2f sentence_len=%d query_prev_sentence=%s (visual bank frozen)",
            int(getattr(train_config.model, "memory_v5_write_delay_steps", 0)),
            float(train_config.model.memory_v5_write_conf if args.write_conf is None else args.write_conf),
            int(train_config.model.memory_v5_sentence_len),
            bool(getattr(train_config.model, "memory_v5_query_prev_sentence", False)),
        )

    out_norm_stats = dict(norm_stats)

    fast_tokenizer = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)
    pg = fast_tokenizer._paligemma_tokenizer  # noqa: SLF001
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    configured_delay = getattr(train_config.model, "simulated_delay", None)
    simulated_delay = None if configured_delay is None else int(configured_delay)
    metadata = _build_server_metadata(train_config, data_config, simulated_delay=simulated_delay)
    metadata["fast_decode"] = bool(args.fast_decode)
    return MemoryPolicy(
        model,
        decode_tokenizer=pg,
        stop_token=stop_token,
        max_decode_steps=max_decode_steps,
        num_steps=args.num_steps,
        zero_read=args.zero_read,
        force_subtask=args.force_subtask,
        write_conf=args.write_conf,
        log_steps=args.log_steps,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        raw_action_dim=int(np.asarray(norm_stats["actions"].mean).shape[-1]),
        simulated_delay=simulated_delay,
        fast_decode=args.fast_decode,
        fast_tokenizer=fast_tokenizer,
        transforms=[
            # BuildMemorySequence is the dataset-side sequence builder; live single-frame
            # observations pass through it untouched (no "frame_index"), so no filtering needed.
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(out_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=metadata,
    )


def _warmup(policy: MemoryPolicy, *, prompt: str, raw_action_dim: int, action_horizon: int) -> None:
    """Compile every request shape the robot client uses (plain and RTC-prefixed), then reset."""
    rng = np.random.default_rng(0)
    example = {
        "observation/state": rng.random(raw_action_dim).astype(np.float32),
        "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": prompt,
    }
    history = int(policy.metadata.get("image_history_frames", 0))
    if history > 0:  # 0920_v0: the request shape with the past front frames (the first tick pads them all)
        for i in range(history):
            example[f"observation/history_{i}"] = example["observation/image"]
        example["observation/history_valid"] = np.zeros((history,), dtype=bool)
    started = time.monotonic()
    first = policy.infer(dict(example))
    logging.info(
        "warmup: plain request compiled + ran in %.1f s (subtask %r)", time.monotonic() - started, first["subtask"]
    )
    if policy._simulated_delay is not None:  # noqa: SLF001
        started = time.monotonic()
        policy.infer(
            {
                **example,
                "action_prefix": {
                    "actions": np.asarray(first["actions"], dtype=np.float32)[:action_horizon],
                    "delay": policy._simulated_delay,  # noqa: SLF001
                    "prefix_length": min(action_horizon, policy._simulated_delay + 10),  # noqa: SLF001
                },
            }
        )
        logging.info("warmup: RTC-prefixed request compiled + ran in %.1f s", time.monotonic() - started)
    policy.infer({"reset_memory": True})
    logging.info("warmup done; memory reset")


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.warmup:
        train_config = _config.get_config(args.config)
        prompt = (
            "find the banana"
            if getattr(train_config.model, "memory_v35_enabled", False)
            else "find the bin with banana"
        )
        _warmup(
            policy,
            prompt=prompt,
            raw_action_dim=policy._raw_action_dim,  # noqa: SLF001
            action_horizon=policy._action_horizon,  # noqa: SLF001
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
