"""CPU check of the 0920_v0 configs: instantiate, build the data config, tokenize the longest prompt without the slot."""
import json, pathlib, sys
from openpi.training import config as _config
NAMES = sys.argv[1:] or ["pi05_robomme_0920_v0", "pi05_robomme_0920_v0_smoke"]  # argv: config names (BinFill: pi05_robomme_0920_binfill_v0)
for name in NAMES:
    c = _config.get_config(name)
    m = c.model
    print(name, "| steps", c.num_train_steps, "| batch", c.batch_size, "fsdp", c.fsdp_devices, "| label sched", c.label_write_schedule_steps,
          "| wandb", c.wandb_enabled, c.project_name)
    print("  model: horizon", m.action_horizon, "max_token_len", m.max_token_len, "seq", m.memory_seq_steps, "block", m.memory_block_steps,
          "slot", m.prompt_slot_len, "| v0 read", m.memory_v0920_input_read, "hist", m.memory_v0920_history_frames, m.memory_v0920_history_pool,
          m.memory_v0920_history_dropout, "| pointer", m.memory_v6_pointer_read, "no_visual", m.memory_v7_no_visual_block,
          "| write_every", m.memory_v7_write_every_step, "conf", m.memory_v5_write_conf, "debounce", m.memory_v7_write_debounce_steps,
          "oracle", m.memory_v5_oracle_writes, "| token_writes", m.memory_v6_token_writes, "whiten", m.memory_v6_whiten_keys,
          "| bank", m.memory_semantic.hidden_dims, m.memory_semantic.alpha_step, m.memory_semantic.delta_rate)
    d = c.data.create(c.assets_dirs, m)
    print("  data: stride", d.memory_stride_frames, "buckets", d.memory_sequence_buckets, "slice", d.memory_slice_prob, "crit", d.memory_critical_prob,
          "pad", d.memory_critical_start_pad, "power", d.memory_start_history_power, "| img hist", d.memory_image_history_frames,
          d.memory_image_history_stride, d.memory_image_history_key)
    structure = d.repack_transforms.inputs[0].structure
    print("  repack keys:", sorted(structure))
    inputs = [type(t).__name__ for t in d.data_transforms.inputs]
    print("  data inputs:", inputs)
    ri = [t for t in d.data_transforms.inputs if type(t).__name__ == "RobommeInputs"][0]
    print("  RobommeInputs history", ri.history_frames, "drop blank", ri.drop_blank_camera)
    print("  sets: evidence", len(d.evidence_subtasks), "decision", len(d.memory_required_subtasks), "| vocab", len(d.memory_subtask_vocab), "| repo", d.repo_id, "| root", d.lerobot_dataset_root)
    print("  labels:", d.memory_v5_subtask_labels_path, "| manifest:", d.memory_episode_manifest_path)
    print("  weight loader:", type(c.weight_loader).__name__, getattr(c.weight_loader, "params_path", None) or getattr(c.weight_loader, "checkpoint_path", None))
    spec_obs, _ = m.inputs_spec(batch_size=1)
    print("  inputs_spec images:", sorted(spec_obs.images), spec_obs.images["base_0_rgb"].shape)
# prompt length without the slot over all goals
from openpi.models import tokenizer as _tok
import numpy as np
c = _config.get_config("pi05_robomme_0920_v0")
root = pathlib.Path("/iris/u/kewalk/memory_project_0920")
prompts = json.loads((root / "robomme/data/lerobot/PickXtimes_official/meta/episode_prompts.json").read_text())
model_tf = [t for t in c.data.create(c.assets_dirs, c.model).model_transforms.inputs]
print("  model transforms:", [type(t).__name__ for t in model_tf])
tk = [t for t in model_tf if "Tokenize" in type(t).__name__][0]
worst = 0
for goal in set(prompts.values()):
    out = tk({"prompt": goal, "state": np.full((40, 8), -0.999, dtype=np.float32), "actions": np.zeros((40, 40, 8), dtype=np.float32), "subtask": ["pick up the green cube for the first time"] * 40,
              "seq_step_mask": np.ones(40, bool)})
    n = int(np.asarray(out["tokenized_prompt_mask"]).sum(-1).max())
    worst = max(worst, n)
print("  longest prompt tokens without slot:", worst, "of", c.model.max_token_len)
