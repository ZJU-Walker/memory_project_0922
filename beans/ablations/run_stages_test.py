"""Exercise the real SNAP-MLP3 wrapper/chain with a CPU-only fake stage payload."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("external_storage", [False, True])
@pytest.mark.parametrize("explicit_run_name", [False, True])
@pytest.mark.parametrize("accum", [1, 3])
@pytest.mark.parametrize("row,a_steps,recipe", [
    ("snap_mlp3", 250, "template_slot_snap_mlp3_v1"),
    ("snap_mlp3_a9align", 500, "a9align_snap_mlp3_v1"),
    ("snap_token_mlp3_a9align", 500, "a9align_token_mlp3_v1"),
    ("vis8_mlp3_a9align", 500, "a9align_slot_aux_mlp3_v1"),
    ("state8_mlp3_a9align", 500, "a9align_slot_aux_mlp3_v1"),
    ("vis8s_mlp3_a9align", 500, "a9align_slot_aux_mlp3_v1"),
])
def test_mlp3_chain_owns_checkpoints_resumes_and_guards_batch(tmp_path, smoke, external_storage, explicit_run_name, accum, row, a_steps, recipe):
    scripts = tmp_path / "beans/ablations"
    scripts.mkdir(parents=True)
    source = Path(__file__).parent
    for name in ("run_stages.sh", f"run_{row}.sh"):
        shutil.copyfile(source / name, scripts / name)
    (scripts / "train_slot_stage.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'case "$CFG" in *_A|*_A_smoke) steps=$OPENPI_BEANS_AB_A_STEPS ;; *) steps=$OPENPI_BEANS_AB_STEPS ;; esac\n'
        'printf "%s|%s|%s|%s\\n" "$CFG" "$EXP" "$OPENPI_BEANS_AB_BATCH" "$OPENPI_BEANS_AB_A_PARAMS" >> "$MEMORY_PROJECT_ROOT/stages.txt"\n'
        'mkdir -p "${OPENPI_BEANS_AB_CHECKPOINT_ROOT:-$MEMORY_PROJECT_ROOT/beans/checkpoints}/$CFG/$EXP/$steps/params"\n'
    )
    base = tmp_path / "beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params"
    base.mkdir(parents=True)
    default_run = "token_mlp3_a9align" if row == "snap_token_mlp3_a9align" else f"slot_{row}"
    selected_run = f"{default_run}_b12" if explicit_run_name else default_run
    env = dict(os.environ, MEMORY_PROJECT_ROOT=str(tmp_path), OPENPI_BEANS_BASE_PARAMS=str(base),
               BATCH="12", ACCUM=str(accum), STEPS="3000")
    env.pop("RUN_NAME", None)
    if explicit_run_name:
        env["RUN_NAME"] = selected_run
    env.pop("A_STEPS", None)  # verify the row's actual default A length
    env.pop("OPENPI_BEANS_AB_CHECKPOINT_ROOT", None)
    checkpoint_root = tmp_path / ("external_checkpoints" if external_storage else "beans/checkpoints")
    if external_storage:
        env["OPENPI_BEANS_AB_CHECKPOINT_ROOT"] = str(checkpoint_root)
    cmd = ["bash", str(scripts / f"run_{row}.sh"), *(('smoke',) if smoke else ())]
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=15)
    suffix = "_smoke" if smoke else ""
    run = ("smoke_" if smoke else "") + selected_run
    a_step, b_step = (2, 2) if smoke else (a_steps, 3000)
    a_config = f"pi05_yam_beans0922_ab_{row}_A{suffix}"
    b_config = f"pi05_yam_beans0922_ab_{row}{suffix}"
    a_params = checkpoint_root / f"{a_config}/{run}_A/{a_step}/params"
    assert a_params.is_dir()
    assert (checkpoint_root / f"{b_config}/{run}_B/{b_step}/params").is_dir()
    expected = [f"{a_config}|{run}_A|12|{a_params}", f"{b_config}|{run}_B|12|{a_params}"]
    assert (tmp_path / "stages.txt").read_text().splitlines() == expected
    for stage in ("A", "B"):
        assert (scripts / f"logs/{run}_{stage}.recipe").read_text().startswith(
            f"{recipe} row={row} stage={stage} batch=12 accum={accum}"
        )
    # Completed stages are skipped, never overwritten or imported from vis8/linear SNAP.
    result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=15)
    assert "A already completed" in result.stdout and "B already completed" in result.stdout
    assert (tmp_path / "stages.txt").read_text().splitlines() == expected
    result = subprocess.run(cmd, env=dict(env, BATCH="8"), capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and "Recipe changed" in result.stdout
    result = subprocess.run(cmd, env=dict(env, ACCUM=str(3 if accum == 1 else 1)), capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and "Recipe changed" in result.stdout
    if external_storage:
        result = subprocess.run(cmd, env=dict(env, OPENPI_BEANS_AB_CHECKPOINT_ROOT=str(tmp_path / "different_storage")),
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 2 and "Recipe changed" in result.stdout
