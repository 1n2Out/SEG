"""Offline figure checks only: no model inference, backward or optimization."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from brats.data import MODALITIES, PREPROCESS_VERSION, digest, write_json
from brats.plotting import (configure, load_case, patient_statistics,
                            read_history, select_slices, slice_view)
from plot_brats import main, parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_axes_auto_selection_crop_and_empty_region():
    label = np.zeros((4, 6, 8), np.uint8)
    label[1:3, 2:5, 4] = 4
    assert select_slices(label, "axial") == [4]
    assert select_slices(label, "sagittal") == [1]
    assert select_slices(label, "axial", indices=[0, 7], radius=1) == [0, 1, 6, 7]
    assert slice_view(label, "axial", 4).shape == (6, 4)
    assert np.all(slice_view(label, "axial", 4, [1, 2, 3, 5]) == 4)
    assert select_slices(np.zeros_like(label), "coronal", "ET") == [3]
    with pytest.raises(ValueError, match="outside"):
        select_slices(label, "axial", indices=[8])
    with pytest.raises(ValueError, match="Crop"):
        slice_view(label, "axial", 4, [0, 0, 5, 6])


def test_statistics_nested_regions_physical_units_and_negative_slices():
    label = np.zeros((2, 3, 4), np.uint8)
    label[0, 0, 0], label[0, 1, 1], label[1, 2, 1] = 1, 2, 4
    # Meter-unit affine: 2x3x4 mm voxels. Counts are regions, not exclusive labels.
    meta = {"affine": np.diag([0.002, 0.003, 0.004, 1]).tolist(), "spacing_xyz_mm": [2, 3, 4]}
    row, slices = patient_statistics("case", label, meta)
    assert (row["WT_voxels"], row["TC_voxels"], row["ET_voxels"]) == (3, 2, 1)
    assert row["WT_ml"] == pytest.approx(3 * 24 / 1000)
    assert row["WT_positive_slices"] == 2 and row["ET_positive_slices"] == 1
    assert len(slices) == 4 and slices[3]["WT_voxels"] == 0
    assert sum(row[f"label_{v}_voxels"] for v in (0, 1, 2, 4)) == row["total_voxels"]


def test_live_history_legacy_fallback_and_malformed_rows(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_bytes(b'{"epoch":1,"train_loss":1.5,"val_mean_dice":0.5}\n{"epoch":')
    write_json(tmp_path / "validation/epoch_001/summary.json",
               {"regions": {r: {"dice": {"mean": 0.5}} for r in ("WT", "TC", "ET")}})
    with pytest.warns(UserWarning, match="unfinished"):
        records, sha = read_history(tmp_path)
    assert len(records) == 1 and records[0]["val_dice"]["ET"] == 0.5
    assert "train_ce_loss" not in records[0] and len(sha) == 64
    path.write_text('{"epoch":1}\n{"epoch":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_history(tmp_path)
    path.write_text('{"epoch":1}\nbroken\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        read_history(tmp_path)


@pytest.fixture
def plotting_data(tmp_path):
    cases = {}
    for i in range(3):
        case_id = f"case{i}"
        cases[case_id] = {"id": case_id, "label": "unused", "images": {m: "unused" for m in MODALITIES}}
        directory = tmp_path / "cache" / case_id
        directory.mkdir(parents=True)
        label = np.zeros((8, 4, 6), np.uint8)
        label[4, 1:3, 2:5] = 4
        np.save(directory / "label.npy", label)
        np.save(directory / "image.npy", np.ones((8, 4, 4, 6), np.float16))
        write_json(directory / "metadata.json", {"preprocess_version": PREPROCESS_VERSION,
                   "case_id": case_id, "modalities": list(MODALITIES), "shape_xyz": [4, 6, 8],
                   "spacing_xyz_mm": [1, 1, 1], "spatial_unit": "mm", "affine": np.eye(4).tolist()})
    manifest = {"version": 2, "purpose": "training", "modalities": list(MODALITIES),
                "splits": {"train": ["case0"], "val": ["case1"], "test": ["case2"]}, "cases": cases}
    manifest["sha256"] = digest(manifest)
    write_json(tmp_path / "manifest.json", manifest)
    return tmp_path


def test_case_and_stats_cli_exports(plotting_data):
    root = plotting_data
    common = ["--manifest", str(root / "manifest.json"), "--cache-dir", str(root / "cache"),
              "--formats", "png", "pdf", "svg", "--dpi", "60"]
    main(["case", *common, "--output-dir", str(root / "case_figures")])
    output = root / "case_figures/case0"
    with np.load(output / "case_slices.npz") as arrays:
        assert arrays["axial_004_images"].shape == (4, 6, 4)
        assert arrays["axial_004_labels"].max() == 4
    for extension in ("png", "pdf", "svg"):
        assert (output / f"axial_004.{extension}").stat().st_size > 100
    main(["stats", *common, "--output-dir", str(root / "stats_figures")])
    meta = json.loads((root / "stats_figures/stats_metadata.json").read_text())
    assert meta["case_ids"] == ["case0"] and meta["partial"] is False
    with (root / "stats_figures/class_balance.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(float(r["percent_whole_grid"]) for r in rows) == pytest.approx(100)
    # Training-mapped label 3 must not be accepted as an original BraTS label.
    labels = np.load(root / "cache/case0/label.npy")
    labels[0, 0, 0] = 3
    np.save(root / "cache/case0/label.npy", labels)
    with pytest.raises(ValueError, match="Invalid BraTS labels"):
        load_case({"id": "case0", "label": "unused"}, root, root / "cache")


def test_curve_cli_exports_observed_epochs_and_missing_legacy_components(tmp_path):
    run = tmp_path / "fixture_run"
    run.mkdir()
    records = [{"epoch": 1, "train_loss": 2, "train_seconds": 10, "val_mean_dice": 0.5},
               {"epoch": 2, "train_loss": 1.5, "train_seconds": 11},
               {"epoch": 5, "train_loss": 1, "train_seconds": 12, "train_ce_loss": 0.4,
                "train_dice_loss": 0.6, "val_mean_dice": 0.7,
                "val_dice": {r: 0.7 for r in ("WT", "TC", "ET")}, "early_stopping": {"stopped": True}}]
    (run / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    output = tmp_path / "figures"
    main(["curves", "--work-dirs", str(run), "--names", "Fixture only", "--formats", "png",
          "--dpi", "60", "--output-dir", str(output)])
    with (output / "curve_data.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["epoch"] for r in rows] == ["1", "2", "5"]
    assert rows[1]["val_mean_dice"] == "" and rows[0]["train_ce_loss"] == ""
    meta = json.loads((output / "curves_metadata.json").read_text())
    assert meta["runs"][0]["best_observed_epoch"] == 5 and meta["runs"][0]["early_stopped"]
    assert len(list(output.glob("*.png"))) == 4


def test_loss_components_preserve_objective_forward_only():
    torch = pytest.importorskip("torch")
    from brats.losses import BraTSCEDiceLoss
    with torch.no_grad():
        logits = torch.linspace(-2, 2, 2 * 4 * 6 * 6).reshape(2, 4, 6, 6)
        target = torch.arange(72).reshape(2, 6, 6) % 4
        total, components = BraTSCEDiceLoss()(logits, target, return_components=True)
        assert torch.equal(total, BraTSCEDiceLoss()(logits, target))
        assert torch.equal(total, components["ce"] + components["dice"])
        assert torch.equal(components["ce"], torch.nn.functional.cross_entropy(logits, target))
        assert all(not v.requires_grad for v in components.values())


def test_plotting_config_and_cli_defaults():
    style = configure(ROOT / "configs/plotting.json")
    assert style["formats"] == ["png", "pdf", "svg"]
    assert parse_args(["stats"]).split == "train"
    with pytest.raises(ValueError, match="dpi"):
        configure(ROOT / "configs/plotting.json", dpi=0)
