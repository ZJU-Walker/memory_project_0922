"""Record the benchmark's own motion planner (the "expert") solving PickXtimes episodes frame by frame, for offline
memory evals on scenes the model never trained on (all 100 released demos are the train split; val has no demos).

    record_expert_val.py --split val --episodes 1,4,0,3,15 --out <root>/robomme/expert_val

Writes <out>/PickXtimes_val_ep<NNN>/{frames.npz (front, wrist uint8 [T,256,256,3]; joint [T,7]; gripper [T,2]),
meta.json (goal, seed, per-frame online labels and status), preview.mp4}. The planner reads privileged simulator state,
so these recordings are diagnostic data only, never a benchmark number. Labels switch at the planner's task boundaries,
exactly as in the released training demos (DemonstrationWrapper.get_demonstration_trajectory).
"""
import argparse
import json
import pathlib
import time

import numpy as np

from common import array


def _col(ob, key):
    v = ob[key]
    return array(v[-1] if isinstance(v, list) else v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="PickXtimes")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--episodes", default="1,4,0,3,15")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--max-steps", type=int, default=1300)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    builder = BenchmarkEnvBuilder(args.task, dataset=args.split, action_space="joint_angle", max_steps=args.max_steps)
    for ep in [int(x) for x in args.episodes.split(",")]:
        d = args.out / f"{args.task}_{args.split}_ep{ep:03d}"
        if (d / "meta.json").exists():
            print(f"skip {d} (exists)", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        env = builder.make_env_for_episode(ep, max_steps=args.max_steps)
        try:
            obs0, info0 = env.reset()
            goal = info0["task_goal"][0]
            seed, difficulty = builder.resolve_episode(ep)
            tasks = env.unwrapped.task_list
            for t in tasks:
                t["demonstration"] = True  # the released val scenes ship with the planner disabled: the policy must act
            print(f"ep {ep}: seed={seed} difficulty={difficulty} tasks={len(tasks)} goal={goal!r}", flush=True)
            obs_b, _reward, _term, _trunc, info_b = env.get_demonstration_trajectory()
            fronts, wrists, joints, grips = [_col(obs0, "front_rgb_list")], [_col(obs0, "wrist_rgb_list")], [_col(obs0, "joint_state_list")], [_col(obs0, "gripper_state_list")]
            labels, grounded, status = [info0.get("simple_subgoal_online")], [info0.get("grounded_subgoal_online")], [str(info0.get("status", "ongoing"))]
            n = len(obs_b.get("front_rgb_list", []))
            for i in range(n):
                fronts.append(array(obs_b["front_rgb_list"][i])); wrists.append(array(obs_b["wrist_rgb_list"][i]))
                joints.append(array(obs_b["joint_state_list"][i])); grips.append(array(obs_b["gripper_state_list"][i]))
                labels.append(info_b["simple_subgoal_online"][i]); grounded.append((info_b.get("grounded_subgoal_online") or [None] * n)[i])
                status.append(str(info_b["status"][i]))
            success = bool(getattr(env, "episode_success", False)) or status[-1] == "success"
        finally:
            env.close()
        front = np.stack(fronts).astype(np.uint8); wrist = np.stack(wrists).astype(np.uint8)
        joint = np.stack(joints).astype(np.float32).reshape(len(fronts), -1)[:, :7]
        grip = np.stack(grips).astype(np.float32).reshape(len(fronts), -1)[:, :2]
        np.savez_compressed(d / "frames.npz", front=front, wrist=wrist, joint=joint, gripper=grip)
        changes = [(i, l) for i, l in enumerate(labels) if i == 0 or l != labels[i - 1]]
        meta = dict(task=args.task, split=args.split, episode=ep, goal=goal, seed=seed, difficulty=difficulty,
                    num_frames=len(fronts), fps=args.fps, labels=labels, grounded=grounded, status=status,
                    final_status=status[-1], success=success, label_changes=changes, source="planner_expert",
                    record_seconds=round(time.time() - t0, 1), recorded_at=time.time())
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        try:
            import imageio
            with imageio.get_writer(str(d / "preview.mp4"), fps=args.fps, codec="libx264", quality=7, macro_block_size=None) as w:
                for f, wr in zip(front, wrist):
                    w.append_data(np.hstack([f, wr]))
        except Exception as exc:  # preview only
            print(f"preview skipped: {exc}", flush=True)
        print(f"ep {ep}: {len(fronts)} frames, final {status[-1]}, success={success}, {len(changes)} label segments, "
              f"{round(time.time() - t0)}s -> {d}", flush=True)
        for i, l in changes:
            print(f"    frame {i:>4}: {l}", flush=True)


if __name__ == "__main__":
    main()
