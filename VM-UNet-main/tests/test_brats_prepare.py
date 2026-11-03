"""Exercise preparation CLI with synthetic inventories, without NIfTI dependencies."""
import subprocess
import sys
from pathlib import Path

from brats.data import MODALITIES, load_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_brats.py"


def make_case(root, case_id, labeled):
    directory = root / case_id
    directory.mkdir(parents=True)
    for suffix in (*MODALITIES, *(("seg",) if labeled else ())):
        (directory / f"{case_id}_{suffix}.nii").touch()


def run_prepare(root, manifest, *extra):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--data-root", str(root),
                           "--manifest", str(manifest), *extra], capture_output=True, text=True)


def test_training_only_preparation_and_independent_official_preparation(tmp_path):
    root = tmp_path / "raw"
    for i in range(3):
        make_case(root, f"BraTS20_Training_{i:03d}", True)
    training_path = tmp_path / "train.json"
    first = run_prepare(root, training_path)
    assert first.returncode == 0, first.stderr
    training = load_manifest(training_path)
    make_case(root, "BraTS20_Validation_001", False)
    second = run_prepare(root, training_path)
    assert second.returncode == 0, second.stderr
    assert load_manifest(training_path) == training
    official_path = tmp_path / "official.json"
    official = run_prepare(root, official_path, "--official-val")
    assert official.returncode == 0, official.stderr
    assert load_manifest(official_path)["purpose"] == "prediction"


def test_different_split_still_cannot_overwrite_existing_manifest(tmp_path):
    root = tmp_path / "raw"
    for i in range(10):
        make_case(root, f"BraTS20_Training_{i:03d}", True)
    target = tmp_path / "train.json"
    assert run_prepare(root, target).returncode == 0
    original = target.read_bytes()
    changed = run_prepare(root, target, "--seed", "13")
    assert changed.returncode != 0 and "Existing manifest differs" in changed.stderr
    assert target.read_bytes() == original
