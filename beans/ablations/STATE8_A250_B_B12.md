# State-slot: existing A250 -> B3000, batch12 without accumulation

This opt-in entrypoint leaves the standard four-way recipes and all running jobs unchanged.
It starts only `state8_mlp3_a9align` B from the saved **A250** of
`slot_state8_mlp3_a9align_b12_acc3_A`. The entire A parameter tree is restored, with a fresh B optimizer
and B step counter. No KI reinitialization, A rerun, or import of the old A optimizer.
Rerunning the same command resumes the new B's own checkpoint/optimizer; a completed B is skipped.

The effective batch remains12: `ACCUM3` used microbatch4 (one sequence/card), while **ACCUM1** uses
microbatch12 (three sequences/card). This can improve throughput but increases peak GPU memory.
This request follows the user's measurements on Cardinal's four ~94GiB H100s; it is not a fit guarantee
for four 80GB H100s. No automatic batch reduction or fallback occurs.

The B model, learning rate, RTC15, 25/25/50 sampling with pad75, three-layer slot banks, conditioned
layer8 reads, loss weights, and 250-step save cadence stay unchanged. B's outside-scan image encoding
remains enabled. **A250 instead of A500 is an intentional experimental difference**, not a strict
comparison to an A500-trained baseline. The run name and recipe marker record A250 and ACCUM1.

## Stop only the old state-slot chain

On the compute node hosting this run, inside tmux (Cardinal path below; use your own checkout path elsewhere).
First verify that A250 exists. A progress display beyond250 alone is not evidence of a finalized save.
The controller also stops the old launcher so it cannot advance to B. It preserves other runs, servers,
1GB keep-alives, the Slurm allocation, checkpoints and logs. Progress after the saved A250 is not used.

```bash
cd /fs/scratch/PAS2099/memory_project_beans0922
(
  set -euo pipefail
  test -d beans/checkpoints/pi05_yam_beans0922_ab_state8_mlp3_a9align_A/slot_state8_mlp3_a9align_b12_acc3_A/250/params
  bash beans/ablations/ablation_ctl.sh status state8_mlp3_a9align --run-name slot_state8_mlp3_a9align_b12_acc3
  bash beans/ablations/ablation_ctl.sh stop state8_mlp3_a9align --run-name slot_state8_mlp3_a9align_b12_acc3
  bash beans/ablations/ablation_ctl.sh status state8_mlp3_a9align --run-name slot_state8_mlp3_a9align_b12_acc3
)
git pull --ff-only origin main
nvidia-smi
```

The final status should show zero matching processes. Do not kill any other GPU users. No archive/delete/rename is needed.

## Launch B directly

If needed, set the W&B key in this same shell without displaying it:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
```

```bash
(
  set -euo pipefail
  export JOB="${SLURM_JOB_ID:-}" GRES=4 CPUS=24 GPUS=0,1,2,3 WORKERS=4
  export B_RUN_NAME=slot_state8_mlp3_a9align_A250_b12_acc1
  export A250_PARAMS="$PWD/beans/checkpoints/pi05_yam_beans0922_ab_state8_mlp3_a9align_A/slot_state8_mlp3_a9align_b12_acc3_A/250/params"
  unset OPENPI_BEANS_AB_CHECKPOINT_ROOT
  bash beans/ablations/run_state8_mlp3_a9align_b_from_a250.sh check
  mkdir -p beans/ablations/logs
  bash beans/ablations/run_state8_mlp3_a9align_b_from_a250.sh \
    2>&1 | tee -a "beans/ablations/logs/run_${B_RUN_NAME}.out"
)
```

This explicitly fixes batch12/ACCUM1 even if the old shell still exports `ACCUM=3`, `BATCH`, `A_STEPS`,
`STEPS`, `RUN_NAME` or `OPENPI_BEANS_AB_A_PARAMS`. No A or automatic smoke runs are launched. Optional `smoke`
mode tests only B to checkpoint2 from the real A250, in a separate namespace with W&B disabled.
For a different source/storage location, set `A250_PARAMS` / `OPENPI_BEANS_AB_CHECKPOINT_ROOT` explicitly;
the source must retain the state8 A config and `250/params` directory layout. Changed source/signature requires a new `B_RUN_NAME`.

From another tmux window:

```bash
tail -n 50 -F beans/ablations/logs/train_slot_state8_mlp3_a9align_A250_b12_acc1_B.log
```

W&B experiment: `slot_state8_mlp3_a9align_A250_b12_acc1_B` (same `beans0922_ablation` project).
Checkpoints: `beans/checkpoints/pi05_yam_beans0922_ab_state8_mlp3_a9align/slot_state8_mlp3_a9align_A250_b12_acc1_B/`.
Detach tmux with Ctrl+b,d; the Slurm allocation must remain valid.
