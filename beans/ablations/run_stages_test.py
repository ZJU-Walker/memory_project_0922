"""Exercise the real SNAP-MLP3 wrapper/chain with a CPU-only fake stage payload."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("smoke", [False, True])
def test_mlp3_chain_owns_checkpoints_resumes_and_guards_batch(tmp_path, smoke):
    scripts = tmp_path / "beans/ablations"
    scripts.mkdir(parents=True)
    source = Path(__file__).parent
    for name in ("run_stages.sh", "run_snap_mlp3.sh"):
        shutil.copyfile(source / name, scripts / name)
    (scripts / "train_slot_stage.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'case "$CFG" in *_A|*_A_smoke) steps=$OPENPI_BEANS_AB_A_STEPS ;; *) steps=$OPENPI_BEANS_AB_STEPS ;; esac\n'
        'printf "%s|%s|%s|%s\\n" "$CFG" "$EXP" "$OPENPI_BEANS_AB_BATCH" "$OPENPI_BEANS_AB_A_PARAMS" >> "$MEMORY_PROJECT_ROOT/stages.txt"\n'
        'mkdir -p "$MEMORY_PROJECT_ROOT/beans/checkpoints/$CFG/$EXP/$steps/params"\n'
    )
    base = tmp_path / "beans/checkpoints/pi05_yam_beans0922_base/beans0922_base/10000/params"
    base.mkdir(parents=True)
    env = dict(os.environ, MEMORY_PROJECT_ROOT=str(tmp_path), OPENPI_BEANS_BASE_PARAMS=str(base),
               RUN_NAME="slot_snap_mlp3_b12", BATCH="12", ACCUM="1", A_STEPS="250", STEPS="3000")
    cmd = ["bash", str(scripts / "run_snap_mlp3.sh"), *(('smoke',) if smoke else ())]
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=15)
    suffix = "_smoke" if smoke else ""
    run = ("smoke_" if smoke else "") + "slot_snap_mlp3_b12"
    a_step, b_step = (2, 2) if smoke else (250, 3000)
    a_config = f"pi05_yam_beans0922_ab_snap_mlp3_A{suffix}"
    b_config = f"pi05_yam_beans0922_ab_snap_mlp3{suffix}"
    a_params = tmp_path / f"beans/checkpoints/{a_config}/{run}_A/{a_step}/params"
    assert a_params.is_dir()
    assert (tmp_path / f"beans/checkpoints/{b_config}/{run}_B/{b_step}/params").is_dir()
    expected = [f"{a_config}|{run}_A|12|{a_params}", f"{b_config}|{run}_B|12|{a_params}"]
    assert (tmp_path / "stages.txt").read_text().splitlines() == expected
    for stage in ("A", "B"):
        assert (scripts / f"logs/{run}_{stage}.recipe").read_text().startswith(
            f"template_slot_snap_mlp3_v1 row=snap_mlp3 stage={stage} batch=12 accum=1"
        )
    # Completed stages are skipped, never overwritten or imported from vis8/linear SNAP.
    result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=15)
    assert "A already completed" in result.stdout and "B already completed" in result.stdout
    assert (tmp_path / "stages.txt").read_text().splitlines() == expected
    result = subprocess.run(cmd, env=dict(env, BATCH="8"), capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and "Recipe changed" in result.stdout
