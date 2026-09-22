#!/usr/bin/env bash
# CPU checks of pi05_yam_beans_0920_v1 (ON iris-hgx-1, no GPU): unit tests, config print, one real loader batch of the smoke config.
set -u
ROOT=/iris/u/kewalk/memory_project_0920; cd $ROOT/openpi || exit 2
export HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
source cluster_robomme/env.sh >/dev/null 2>&1
find src/openpi -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
echo "== pytest $(date +%H:%M:%S)"; .venv/bin/python -m pytest -q src/openpi/training/beans_0920_test.py src/openpi/training/robomme_0920_test.py -p no:cacheprovider 2>&1 | tail -4
echo "== config print"; .venv/bin/python - <<'PY' 2>&1 | grep -v "^WARNING\|^I0\|^W0" | cut -c1-300
from openpi.training import config as C
c = C.get_config("pi05_yam_beans_0920_v1"); m = c.model; d = c.data.base_config
print("model: horizon", m.action_horizon, "max_tok", m.max_token_len, "seq", m.memory_seq_steps, "block", m.memory_block_steps, "delay", m.simulated_delay, "slot", m.prompt_slot_len,
      "| input read", m.memory_v0920_input_read, "hist", m.memory_v0920_history_frames, "pointer", m.memory_v6_pointer_read, "no_visual", m.memory_v7_no_visual_block, "q_prev", m.memory_v5_query_prev_sentence,
      "| write_every", m.memory_v7_write_every_step, "conf", m.memory_v5_write_conf, "oracle", m.memory_v5_oracle_writes, "own_label", m.memory_v5_own_commit_label_content, "| onset", m.memory_v7_onset_ce_weight, "state_mask", m.memory_state_mask_prob, "prefill", m.memory_v5_prefill_max)
print("data: stride", d.memory_stride_frames, "buckets", d.memory_sequence_buckets, "slice", d.memory_slice_prob, "crit", d.memory_critical_prob, "pad", d.memory_critical_start_pad, "minslice", d.memory_min_slice_steps, "| vocab", len(d.memory_subtask_vocab), "| root", d.lerobot_dataset_root)
print("labels:", d.memory_v5_subtask_labels_path, "| manifest:", d.memory_episode_manifest_path)
print("train: batch", c.batch_size, "fsdp", c.fsdp_devices, "steps", c.num_train_steps, "ramp", c.label_write_schedule_steps, "lr", c.lr_schedule, "save", c.save_interval, "keep", c.keep_period, "max_keep", c.checkpoint_max_to_keep, "workers", c.num_workers, "wandb", c.wandb_enabled, c.project_name)
print("loader:", c.weight_loader.params_path, "| ckpts:", c.checkpoint_base_dir, "| assets:", c.data.assets)
o, a = m.inputs_spec(batch_size=1); print("inputs_spec images", sorted(o.images), "actions", a.shape)
PY
echo "== loader probe smoke (one batch) $(date +%H:%M:%S)"; srun --jobid=${JOB:-17489557} --overlap --nodes=1 --ntasks=1 --cpus-per-task=12 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 .venv/bin/python $ROOT/robomme/logs/probe_0920_data.py pi05_yam_beans_0920_v1_smoke 2>&1 | grep -E "loader built|images|actions|mask|prompt tokens|Traceback|Error|error" | head -14 | cut -c1-300
echo "== done $(date +%H:%M:%S)"
