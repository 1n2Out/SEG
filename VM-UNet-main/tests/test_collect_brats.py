"""collect_brats exports validation results from a synthetic run directory."""
import csv
import json

import pytest

import collect_brats

FIELDS = ["case_id"] + [f"{region}_{name}" for region in collect_brats.REGIONS
                        for name in ("dice", "hd95_mm", "precision", "recall", "specificity",
                                     "target_voxels", "prediction_voxels")]


def write_epoch(directory, mean, et_rows):
    directory.mkdir(parents=True)
    regions = {region: {"dice": {"mean": mean}, "hd95_mm": {"mean": 5.0}} for region in collect_brats.REGIONS}
    (directory / "summary.json").write_text(json.dumps(
        {"mean_dice": mean, "mean_hd95_mm": 5.0, "regions": regions}), encoding="utf-8")
    with (directory / "per_patient.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        for index, (target, prediction, dice) in enumerate(et_rows):
            row = {field: 1 for field in FIELDS}
            row.update(case_id=f"case_{index}", ET_target_voxels=target,
                       ET_prediction_voxels=prediction, ET_dice=dice)
            writer.writerow(row)


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "results" / "demo_1102_1734"
    history = [{"epoch": 1, "train_seconds": 1800, "val_mean_dice": 0.70, "best_epoch": 1},
               {"epoch": 2, "train_seconds": 1800, "best_epoch": 1},
               {"epoch": 3, "train_seconds": 3600, "val_mean_dice": 0.80, "best_epoch": 3}]
    run.mkdir(parents=True)
    (run / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "epochs": 3, "batch_size": 16, "learning_rate": 1e-4, "pretrained": "../../package/w.pth",
        "detail_fusion": {"mode": "gated"}}), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"parameters": 10, "git": {"commit": "abc"}}), encoding="utf-8")
    write_epoch(run / "validation" / "epoch_001", 0.70, [(0, 0, 1.0), (10, 8, 0.5)])
    # Epoch 3: one of two empty-ET patients receives a false positive.
    write_epoch(run / "validation" / "epoch_003", 0.80, [(0, 0, 1.0), (0, 5, 0.0), (10, 8, 0.6), (20, 18, 0.8)])
    test = run / "test_1103_0902" / "evaluation_test"
    test.mkdir(parents=True)
    (test / "summary.json").write_text(json.dumps({"mean_dice": 0.99}), encoding="utf-8")
    (test / "per_patient.csv").write_text("case_id\n", encoding="utf-8")
    return run


def test_collects_best_epoch_validation_without_test(run_dir, tmp_path):
    out = tmp_path / "experiments"
    summary = collect_brats.collect(run_dir, "EXP-001", "Baseline", out)
    assert summary["selected_epoch"] == 3 and summary["selection"] == "best"
    assert summary["epochs_completed"] == 3 and summary["train_hours"] == 2.0
    assert summary["git"] == {"commit": "abc"} and summary["parameters"] == 10
    assert summary["config"]["pretrained"] == "w.pth"
    assert summary["config"]["extension"] == {"detail_fusion": {"mode": "gated"}}
    val = summary["val"]
    assert val["mean_dice"] == 0.80 and val["ET"]["dice"] == 0.80
    assert val["et_empty_false_positive"] == "1/2"
    assert val["et_nonempty_dice"] == pytest.approx(0.7)
    target = out / "EXP-001"
    assert sorted(path.name for path in target.iterdir()) == [
        "config.json", "history.jsonl", "summary.json", "val_per_patient.csv"]
    index = (out / "index.md").read_text(encoding="utf-8")
    assert index.startswith(collect_brats.INDEX_HEADER)
    assert "[EXP-001](EXP-001/report.md)" in index and "| 80.00 |" in index and "99" not in index


def test_fixed_epoch_force_report_and_delta(run_dir, tmp_path):
    out = tmp_path / "experiments"
    collect_brats.collect(run_dir, "EXP-001", "Baseline", out)
    (out / "EXP-001" / "report.md").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        collect_brats.collect(run_dir, "EXP-001", "Baseline", out)
    summary = collect_brats.collect(run_dir, "EXP-002", "Variant", out, epoch=1)
    assert summary["selection"] == "fixed" and summary["val"]["et_empty_false_positive"] == "0/1"
    collect_brats.collect(run_dir, "EXP-001", "Baseline", out, force=True)
    assert (out / "EXP-001" / "report.md").read_text(encoding="utf-8") == "keep"
    rows = [line for line in (out / "index.md").read_text(encoding="utf-8").splitlines() if line.startswith("| [EXP")]
    assert rows[0].endswith("| — |") and rows[1].endswith("| -10.00 |")
    with pytest.raises(FileNotFoundError, match="epoch 2"):
        collect_brats.collect(run_dir, "EXP-003", "Missing", out, epoch=2)
    with pytest.raises(ValueError, match="EXP-001"):
        collect_brats.collect(run_dir, "exp1", "Bad", out)


def test_test_results_only_go_to_final(run_dir, tmp_path):
    out = tmp_path / "experiments"
    summary = collect_brats.collect(run_dir, "EXP-001", "Baseline", out, include_test=True)
    final = out / "EXP-001" / "final"
    assert json.loads((final / "test_summary.json").read_text(encoding="utf-8")) == {"mean_dice": 0.99}
    assert (final / "test_per_patient.csv").is_file()
    assert "0.99" not in json.dumps(summary) and "99" not in (out / "index.md").read_text(encoding="utf-8")
