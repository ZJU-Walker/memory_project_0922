#!/usr/bin/env python3
"""Run actual RoboMME episodes and save annotated MP4s plus frame-aligned traces."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time
import traceback
import uuid

import numpy as np

from common import TASKS, HttpPolicyClient, array, jsonable, observation, validate_prediction
from recording import RolloutRecorder


_TGT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
_TGT_ORDINALS = ("first", "second", "third", "fourth", "fifth")


class TgtLabelAdapter:
    """09-17 diagnostic only (never a benchmark number): rewrite the simulator's official online sentence into the
    target-carry label vocabulary ("tgt": 'pick up the red cube, 2 of 3'), so a forced-sentence rollout can feed a
    tgt-trained checkpoint the true sentence. x = number of picks the goal asks for, k = ordinal of the current or
    last pick (the official place sentence carries no ordinal; the tgt place sentence carries its pick's)."""

    def __init__(self, goal):
        self.x = next((v for w, v in _TGT_WORDS.items() if f"{w} times" in goal), 1)
        self.k = 1

    def __call__(self, online):
        s = (online or "").strip().lower()
        if m := re.match(r"^pick up the (\w+) cube for the (\w+) time$", s):
            self.k = _TGT_ORDINALS.index(m.group(2)) + 1
            return f"pick up the {m.group(1)} cube, {self.k} of {self.x}"
        if m := re.match(r"^place the (\w+) cube onto the target$", s):
            return f"place the {m.group(1)} cube onto the target, {self.k} of {self.x}"
        return s  # 'press the button to stop', 'all tasks completed'


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))
    temp.replace(path)


def run_episode(builder, client, args, episode, recorder_class=RolloutRecorder):
    horizon = getattr(args, 'execute_horizon', 0) or args.chunk_size  # actions executed from one plan (memory tick stays --chunk-size)
    run_id = "{}_{}_ep{:03d}_{}_{!s}".format(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                 args.task, episode, args.mode, uuid.uuid4().hex[:6])
    directory = args.output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata = client.metadata if client else {"prediction_source": None}
    seed, difficulty = builder.resolve_episode(episode)
    manifest = dict(schema_version=1, id=run_id, task=args.task, split=args.split, episode=episode,
                    mode=args.mode, started_at=time.time(), state="running", status="ongoing", success=None,
                    environment_seed=seed, difficulty=difficulty, model_seed=args.seed, policy=metadata,
                    max_steps=args.max_steps, step_limit=args.step_limit, action_chunk=args.chunk_size, execute_horizon=horizon,
                    fps=args.fps, control_hz=(args.control_hz or 20), steps=0, frames=0, predictions=0,
                    ground_truth_input=bool(metadata.get("ground_truth_input", False)),
                    diagnostic=args.mode != "policy" or args.split == "train" or bool(args.control_hz) or bool(getattr(args, 'mask_history', False)),
                    history_masked=bool(getattr(args, 'mask_history', False)),
                    oracle_label_version=getattr(args, 'oracle_label_version', 'official') if args.mode == "oracle-subtask" else None,
                    oracle_fixed_sentence=args.oracle_fixed_sentence if args.mode == "oracle-fixed" else None,
                    note="Local task-specialist evaluation; recording FPS does not change simulation timing.")
    write_json(directory / "manifest.json", manifest)
    env, recorder = None, None
    frames = (directory / "frames.jsonl").open("w")
    predictions = (directory / "predictions.jsonl").open("w")
    steps, queries, current, goal = 0, 0, None, ""
    adapter = None  # tgt label adapter for the forced-sentence diagnostic (set once the goal is known)
    try:
        if client:
            client.reset(args.seed)
        env = builder.make_env_for_episode(episode)
        obs, info = env.reset()
        goal = info["task_goal"][0]
        if args.mode == "oracle-subtask" and getattr(args, 'oracle_label_version', 'official') == "tgt":
            adapter = TgtLabelAdapter(goal)
        manifest["goal"] = goal
        recorder = recorder_class(directory, args.task, episode, fps=args.fps,
                         checkpoint="{} update {}{}".format(metadata.get("trained_stage", ""), metadata.get("trained_update", "—"),
                             " | execute {}".format(args.chunk_size) if args.mode == "action-chunk-diagnostic" else ""),
                         mode="base-policy" if metadata.get("prediction_source") == "none" else args.mode)

        def predict(observation_index, phase, source_frame=None):
            nonlocal queries, current
            started = time.monotonic()
            policy_obs = observation(obs, goal, observation_index)
            if args.mode == "oracle-subtask":
                label = info.get("simple_subgoal_online")
                current = client.infer_oracle(policy_obs, adapter(label) if adapter else label)
            elif args.mode == "oracle-fixed":
                # arm diagnostic (09-18): one fixed sentence forced at every tick from tick 0, whatever the scene shows
                current = client.infer_oracle(policy_obs, args.oracle_fixed_sentence)
            else:
                current = client.infer(policy_obs)
            action_plan = validate_prediction(current, horizon)
            row = dict(query=queries, source_frame=recorder.count if source_frame is None else source_frame, step=steps, phase=phase,
                       latency_ms=(time.monotonic() - started) * 1000, **jsonable(current))
            predictions.write(json.dumps(row, allow_nan=False) + "\n")
            predictions.flush()
            queries += 1
            print("{} ep={} step={} prediction={!r} writes={}".format(args.task, episode, steps,
                           current["subtask"], current.get("writes", 0)), flush=True)
            return action_plan

        def record(index, phase, action=None, action_index=None):
            if client and (client.metadata.get("prompt_state_history", 0) or client.metadata.get("image_history_frames", 0)):
                client.observe(obs, index)
            status = str(info.get("status", "ongoing"))
            row = dict(frame=recorder.count, time=recorder.count / args.fps, step=steps, phase=phase,
                       query=queries - 1 if current else None, subtask=(current or {}).get("subtask"),
                       subtask_confidence=(current or {}).get("subtask_confidence"), writes=(current or {}).get("writes", 0),
                       own_subtask=(current or {}).get("own_subtask"), own_confidence=(current or {}).get("own_confidence"),
                       bank=(current or {}).get("bank", []), memory=(current or {}).get("memory", {}),
                       status=status, action=action, action_index=action_index,
                       robot_state=np.r_[array(obs["joint_state_list"][index]).reshape(7),
                                         array(obs["gripper_state_list"][index]).reshape(2)[:1]].tolist())
            # Current online labels correspond only to the last observation, never demo history.
            if index == -1 or index == len(obs["front_rgb_list"]) - 1:
                row["gt_simple"] = info.get("simple_subgoal_online")
                row["gt_detail"] = info.get("grounded_subgoal_online")
            recorder.add(obs["front_rgb_list"][index], obs["wrist_rgb_list"][index], current, steps, phase, status, goal)
            frames.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            frames.flush()

        # Consume the provided conditioning video causally at the trained memory stride.
        # Predictions made here update memory, but their actions are never executed.
        initial_length = len(obs["front_rgb_list"])
        if args.mode in ("oracle-subtask", "oracle-fixed") and initial_length != 1:
            raise ValueError("Oracle diagnostic currently requires a task without a conditioning video")
        for index in range(initial_length - 1):
            if client and index % args.chunk_size == 0:
                predict(index, "conditioning")
            record(index, "conditioning")
        plan = predict(-1, "control") if client else None
        plan_pos = 0
        record(-1, "control")
        done = False
        # Follow the official wrapper's termination/timeout; --step-limit is an explicit preview cap.
        while not done:
            if plan is None:
                state = array(obs["joint_state_list"][-1]).reshape(7)
                actions = np.tile(np.r_[state, 1.0], (args.chunk_size, 1)).astype(np.float32)
            else:
                actions = plan[plan_pos:plan_pos + args.chunk_size]
            for action_index, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                info = info or {}
                done = bool(terminated or truncated or info.get("status") in ("success", "fail", "timeout", "error"))
                if obs is not None:
                    for index in range(len(obs["front_rgb_list"])):
                        record(index, "control", action.tolist(), action_index)
                if info.get("status") == "error":
                    raise RuntimeError(info.get("error_message", "Simulator returned status=error"))
                if args.step_limit and steps >= args.step_limit and not done:
                    manifest["preview_cutoff"] = True
                    done = True
                if done:
                    break
            manifest.update(steps=steps, frames=recorder.count, predictions=queries)
            write_json(directory / "manifest.json", manifest)
            plan_pos += args.chunk_size
            if not done:
                # every query updates the sentence memory (trained tick); its actions replace the plan only when the
                # current plan's execution horizon is used up
                fresh = predict(-1, "control", recorder.count - 1) if client else None
                if fresh is None or plan is None or plan_pos >= horizon:
                    plan, plan_pos = fresh, 0
        status = "preview_cutoff" if manifest.get("preview_cutoff") else str(info.get("status", "unknown"))
        if status in ("ongoing", "unknown") and truncated:
            status = "timeout"
        manifest.update(status=status, state="complete", success=(status == "success") if args.mode == "policy" and status != "preview_cutoff" else None)
        if args.mode in ("oracle-subtask", "oracle-fixed", "action-chunk-diagnostic"):
            manifest["diagnostic_success"] = (status == "success") if status != "preview_cutoff" else None
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="error", status="error", error=str(exc), traceback=traceback.format_exc(), success=None)
        traceback.print_exc()
    finally:
        frames.close()
        predictions.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()
        if recorder:
            try:
                recorder.close()
                manifest.update(video="rollout.mp4" if recorder.count else None,
                                cameras="cameras.mp4" if recorder.count else None,
                                poster="poster.jpg" if recorder.count else None)
            except Exception as exc:
                manifest.update(state="error", status="error", success=None, video_error=str(exc))
        manifest.update(steps=steps, frames=recorder.count if recorder else 0, predictions=queries,
                        finished_at=time.time())
        write_json(directory / "manifest.json", manifest)
        print("SAVED {} status={}".format(directory, manifest["status"]), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", default="0", help="Comma-separated indices, or all")
    parser.add_argument("--policy-url", default="http://127.0.0.1:18767")
    parser.add_argument("--seed", type=int, default=7, help="Model RNG seed; environment seeds stay official")
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--step-limit", type=int, default=0, help="Preview cap; 0 follows official termination")
    parser.add_argument("--chunk-size", type=int, default=5, help="Frames between model queries (= memory tick; training stride 10)")
    parser.add_argument("--execute-horizon", type=int, default=0,
                        help="09-17: execute this many actions of one plan while the model is still queried every --chunk-size frames "
                             "(at most the served plan length: 50 for v1/v2, 40 for v3; a non-multiple of --chunk-size gives a short last chunk, diagnostic); 0 = --chunk-size")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-hz", type=int, default=0,
                        help="09-19 diagnostic: simulator control frequency (benchmark default 20 = 0). E.g. 30 makes 30 steps one simulated "
                             "second; physics runs at the smallest multiple of this at or above 100 Hz. Not benchmark eligible.")
    parser.add_argument("--mode", choices=("policy", "hold-smoke", "oracle-subtask", "oracle-fixed", "action-chunk-diagnostic"), default="policy")
    parser.add_argument("--oracle-fixed-sentence", default="press the button to stop",
                        help="oracle-fixed mode only: the one sentence written to the bank at every tick (arm-reads-the-note diagnostic)")
    parser.add_argument("--oracle-label-version", choices=("official", "tgt"), default="official",
                        help="oracle-subtask mode only: 'tgt' rewrites the official online sentence into the target-carry vocabulary")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "robomme/rollouts")
    parser.add_argument("--mask-history", action="store_true",
                        help="09-21 diagnostic: send the past front frames with every history slot marked invalid (the training dropout condition), so the model runs without its visual history. Not benchmark eligible.")
    args = parser.parse_args()
    horizon = args.execute_horizon or args.chunk_size
    if not 1 <= horizon <= 200:
        parser.error("--execute-horizon must be between 1 and 200 (and at most the served plan length)")
    if horizon % args.chunk_size:
        # 09-19 (user: "execute 50 actions"): a horizon that is not a multiple of the tick runs a short last chunk
        # (e.g. 45..49 for 50/15) and the model is queried when the plan runs out, so one memory tick per plan is
        # shorter than the training stride (5 steps instead of 15). Diagnostic only.
        print("execute horizon {} is not a multiple of the {}-step tick: the last chunk of every plan is short and the "
              "query at plan end comes early".format(horizon, args.chunk_size), flush=True)
    if min(args.max_steps, args.chunk_size, args.fps) < 1 or args.step_limit < 0:
        parser.error("Invalid step count, chunk size or FPS")
    if args.mode == "hold-smoke" and not 1 <= args.step_limit <= 30:
        parser.error("Recording smoke checks require --step-limit between 1 and 30")
    client = HttpPolicyClient(args.policy_url, args.task, args.chunk_size,
                              execution_diagnostic=args.mode == "action-chunk-diagnostic") if args.mode in (
                                  "policy", "oracle-subtask", "oracle-fixed", "action-chunk-diagnostic") else None
    if client and args.mask_history:
        if not client.image_history_frames:
            parser.error("--mask-history needs a checkpoint trained with image history")
        client.mask_history = True
        client.metadata = dict(client.metadata, benchmark_eligible=False, history_masked=True)
        print("history frames MASKED at serving (diagnostic, not benchmark eligible)", flush=True)
    if client and bool(client.metadata.get("diagnostic_oracle_subtask")) != (args.mode in ("oracle-subtask", "oracle-fixed")):
        parser.error("Client/server diagnostic modes do not match")
    if client and horizon > int(client.metadata.get("action_horizon", horizon)):
        # v3 (09-19): plans are action_horizon steps long (50 for v1/v2, 40 for v3); the server says which
        parser.error("--execute-horizon {} exceeds the served plan length {}".format(horizon, client.metadata.get("action_horizon")))
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    if args.control_hz:
        # the builder does not expose env kwargs: inject ManiSkill's sim_config through the module's gym.make
        import math
        import robomme.env_record_wrapper.episode_config_resolver as _ecr
        if not hasattr(_ecr, "gym"):
            parser.error("cannot inject the control frequency: the env builder module has no gym handle")
        sim_hz = args.control_hz * math.ceil(100 / args.control_hz)
        _orig_make = _ecr.gym.make

        def _make_with_freq(env_id, **kw):
            kw = dict(kw)
            kw["sim_config"] = dict(sim_freq=sim_hz, control_freq=args.control_hz)
            return _orig_make(env_id, **kw)

        _ecr.gym.make = _make_with_freq
        print("control frequency {} Hz (physics {} Hz): diagnostic run, not benchmark eligible".format(args.control_hz, sim_hz), flush=True)
    builder = BenchmarkEnvBuilder(args.task, dataset=args.split, action_space="joint_angle", max_steps=args.max_steps)
    episodes = list(range(builder.get_episode_num())) if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    if not episodes or len(set(episodes)) != len(episodes) or any(ep < 0 or ep >= builder.get_episode_num() for ep in episodes):
        parser.error("Episode indices must be unique and present in the selected split")
    results = [run_episode(builder, client, args, ep) for ep in episodes]
    successes = sum(r["success"] is True for r in results)
    scored = sum(r["success"] is not None for r in results)
    print(json.dumps(dict(requested=len(results), completed_with_outcome=scored, successes=successes,
                         success_rate=successes / len(results) if scored == len(results) else None,
                         errors=sum(r["state"] == "error" for r in results))))
    if any(r["state"] == "error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
