"""Run the real B-only launcher/stage with harmless fake GPU/Python commands."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


LAUNCHER = "run_state8_mlp3_a9align_b_from_a250.sh"
ROW = "state8_mlp3_a9align"
RUN = "slot_state8_mlp3_a9align_A250_b12_acc1"
CFG = "pi05_yam_beans0922_ab_state8_mlp3_a9align"


@pytest.fixture
def checkout(tmp_path):
    scripts = tmp_path / "beans/ablations"
    scripts.mkdir(parents=True)
    for name in (LAUNCHER, "train_slot_stage.sh"):
        shutil.copyfile(Path(__file__).parent / name, scripts / name)
    (tmp_path / "openpi").mkdir()
    source = tmp_path / f"beans/checkpoints/{CFG}_A/slot_{ROW}_b12_acc3_A/250/params"
    source.mkdir(parents=True)
    (source / "_METADATA").write_text("source stays unchanged")
    binaries = tmp_path / "fake_bin"
    binaries.mkdir()
    for name, body in {
        "nvidia-smi": "exit 0\n",
        "git": "echo testrevision\n",
        "fake_python": (
            'printf "%s\\n" "$@" > "$MEMORY_PROJECT_ROOT/train_args.txt"\n'
            'printf "%s\\n" "$OPENPI_BEANS_AB_A_PARAMS" "$OPENPI_BEANS_AB_BATCH" '
            '"$OPENPI_BEANS_AB_ACCUM" "$OPENPI_BEANS_AB_A_STEPS" "$OPENPI_BEANS_AB_STEPS" '
            '"$CFG" "$EXP" > "$MEMORY_PROJECT_ROOT/train_env.txt"\n'
            'mkdir -p "${OPENPI_BEANS_AB_CHECKPOINT_ROOT:-$MEMORY_PROJECT_ROOT/beans/checkpoints}/'
            '$CFG/$EXP/${FAKE_SAVE_STEP:-250}/params"\n'
        ),
    }.items():
        path = binaries / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENPI_BEANS_")}
    env.update(MEMORY_PROJECT_ROOT=str(tmp_path), OPENPI_PYTHON=str(binaries / "fake_python"),
               PATH=f"{binaries}:{env['PATH']}", JOB="", GPUS="0,1,2,3", WORKERS="4",
               # Typical inherited old-launcher values must not silently win.
               RUN_NAME=f"slot_{ROW}_b12_acc3", BATCH="8", ACCUM="3", A_STEPS="500", STEPS="99",
               OPENPI_BEANS_AB_A_PARAMS="/incorrect/inherited/source", WANDB="1")
    env.pop("B_RUN_NAME", None)
    env.pop("A250_PARAMS", None)
    return tmp_path, source, env


def run(checkout, mode="train", **overrides):
    root, _, env = checkout
    return subprocess.run(["bash", str(root / "beans/ablations" / LAUNCHER), mode],
                          env=dict(env, **overrides), capture_output=True, text=True, timeout=15)


def test_b_only_forces_batch12_accum1_preserves_a_and_resumes_only_b(checkout):
    root, source, _ = checkout
    result = run(checkout)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (root / "train_env.txt").read_text().splitlines() == [
        str(source), "12", "1", "250", "3000", CFG, RUN + "_B",
    ]
    argv = (root / "train_args.txt").read_text().splitlines()
    assert argv[:2] == ["scripts/train.py", CFG]
    for flag, value in (("--exp-name", RUN + "_B"), ("--batch-size", "12"),
                        ("--gradient-accumulation-steps", "1"), ("--fsdp-devices", "4")):
        assert argv[argv.index(flag) + 1] == value
    assert "--resume" not in argv and "--overwrite" not in argv
    assert "--wandb-enabled" in argv
    assert (source / "_METADATA").read_text() == "source stays unchanged"
    assert not (root / f"beans/checkpoints/{CFG}_A/{RUN}_A").exists()
    marker = root / f"beans/ablations/logs/{RUN}_B.recipe"
    assert "batch=12 accum=1 steps=3000 A=250" in marker.read_text()
    assert f"source={source}" in marker.read_text()
    assert run(checkout).returncode == 0
    assert "--resume" in (root / "train_args.txt").read_text().splitlines()


def test_completed_b_is_skipped(checkout):
    root, _, _ = checkout
    assert run(checkout, FAKE_SAVE_STEP="3000").returncode == 0
    argv = (root / "train_args.txt").read_bytes()
    result = run(checkout)
    assert result.returncode == 0 and "B already completed" in result.stdout
    assert (root / "train_args.txt").read_bytes() == argv


def test_check_does_not_launch_or_create_logs(checkout):
    root, _, _ = checkout
    result = run(checkout, "check")
    assert result.returncode == 0 and "batch=12 accum=1" in result.stdout
    assert not (root / "train_args.txt").exists()
    assert not (root / "beans/ablations/logs").exists()


def test_optional_smoke_is_b_only_from_real_a250(checkout):
    root, source, _ = checkout
    result = run(checkout, "smoke", FAKE_SAVE_STEP="2")
    assert result.returncode == 0, result.stderr
    assert (root / "train_env.txt").read_text().splitlines() == [
        str(source), "12", "1", "250", "2", CFG + "_smoke", "smoke_" + RUN + "_B",
    ]
    assert "--no-wandb-enabled" in (root / "train_args.txt").read_text()


@pytest.mark.parametrize("bad", ["missing", "relative", "wrong_row", "wrong_step", "bad_name"])
def test_invalid_sources_and_names_fail_before_launch(checkout, bad):
    root, _, _ = checkout
    overrides = {}
    if bad == "missing":
        overrides["A250_PARAMS"] = str(root / "missing")
    elif bad == "relative":
        overrides["A250_PARAMS"] = "relative/250/params"
    elif bad == "bad_name":
        overrides["B_RUN_NAME"] = "bad/name"
    else:
        config = CFG if bad == "wrong_step" else "pi05_yam_beans0922_ab_vis8_mlp3_a9align"
        step = 500 if bad == "wrong_step" else 250
        source = root / f"beans/checkpoints/{config}_A/another_A/{step}/params"
        source.mkdir(parents=True)
        overrides["A250_PARAMS"] = str(source)
    assert run(checkout, **overrides).returncode == 2
    assert not (root / "train_args.txt").exists()


def test_unmarked_target_is_never_overwritten(checkout):
    root, _, _ = checkout
    target = root / f"beans/checkpoints/{CFG}/{RUN}_B"
    target.mkdir(parents=True)
    result = run(checkout)
    assert result.returncode == 2 and "Refusing unmarked checkpoint" in result.stdout
    assert not (root / "train_args.txt").exists()


@pytest.mark.parametrize("changed", ["source", "prefill", "storage"])
def test_recipe_change_requires_new_name(checkout, changed):
    root, _, _ = checkout
    assert run(checkout).returncode == 0
    if changed == "source":
        other = root / f"beans/checkpoints/{CFG}_A/another_A/250/params"
        other.mkdir(parents=True)
        overrides = dict(A250_PARAMS=str(other))
    elif changed == "prefill":
        overrides = dict(OPENPI_BEANS_AB_PREFILL_STEPS="160")
    else:
        overrides = dict(OPENPI_BEANS_AB_CHECKPOINT_ROOT=str(root / "another_storage"))
    result = run(checkout, **overrides)
    assert result.returncode == 2 and "Recipe changed" in result.stdout
