"""Row-scoped control tests using harmless sleepers in isolated temporary checkouts."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


CONTROL = Path(__file__).with_name("manage_runs.py")
spec = importlib.util.spec_from_file_location("beans_manage_runs", CONTROL)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.fixture
def sleepers(tmp_path):
    children = {}
    try:
        for row in control.ROWS:
            children[row] = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)", str(tmp_path / "train.py"),
                 f"pi05_yam_beans0922_ab_{row}"],
                cwd=tmp_path, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        yield tmp_path, children
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)


def test_status_does_not_confuse_row_prefixes(sleepers):
    root, children = sleepers
    for row, child in children.items():
        assert set(control.processes(root, row)) == {child.pid}


@pytest.mark.parametrize("row", ["vis8", "vis8s", "snap_mlp3", "snap_mlp3_a9align", "snap_token_mlp3_a9align",
                                "vis8_mlp3_a9align", "state8_mlp3_a9align", "vis8s_mlp3_a9align"])
def test_stop_preserves_other_rows(sleepers, row):
    root, children = sleepers
    subprocess.run(
        [sys.executable, str(CONTROL), "stop", row, "--root", str(root)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    assert children[row].wait(timeout=5) < 0
    assert all(child.poll() is None for name, child in children.items() if name != row)


def test_exact_run_stop_preserves_another_run_of_the_same_row(tmp_path):
    children = {}
    try:
        for run in ("slot_snap_mlp3_b12", "slot_snap_mlp3_b12_other"):
            children[run] = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)", str(tmp_path / "train.py"),
                 "pi05_yam_beans0922_ab_snap_mlp3", "--exp-name", f"{run}_B"],
                cwd=tmp_path, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        chosen = "slot_snap_mlp3_b12"
        assert set(control.processes(tmp_path, "snap_mlp3", chosen)) == {children[chosen].pid}
        subprocess.run([sys.executable, str(CONTROL), "stop", "snap_mlp3", "--run-name", chosen,
                        "--root", str(tmp_path)], check=True, capture_output=True, timeout=15)
        assert children[chosen].wait(timeout=5) < 0
        assert children[chosen + "_other"].poll() is None
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
