"""IDEA-001：细节回注的标签动机与非空 ET 逐患者分析。

输入为固定划分清单、train/val NIfTI 标签和 EXP-001/006/014 的 val best 收集文件。
只读取 train/val 的标签与 val 指标。默认产物在 results/analysis/detail-motivation/。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from brats.data import load_manifest, write_json
from brats.metrics import REGIONS
from brats.plotting import load_label, write_table


EXPERIMENTS = {"baseline": "EXP-001", "concat": "EXP-014", "gated": "EXP-006"}
BLOCK_SIZES = (2, 4)
LABELS = np.array([0, 1, 2, 4], dtype=np.uint8)
AXIAL_NEIGHBORHOOD = np.ones((3, 3, 1), dtype=bool)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 42


def analysis_roi(labels):
    # 240×240 居中补到 256×256 时两侧各补 8 像素，块网格与原图对齐。
    alignment = max(BLOCK_SIZES)
    if any(size % alignment for size in labels.shape[:2]):
        raise ValueError("标签平面尺寸必须能被最大块边长整除")
    foreground = labels != 0
    if not foreground.any():
        return labels[:alignment, :alignment, :1]
    slices = []
    for axis, step in enumerate((alignment, alignment, 1)):
        other_axes = tuple(other for other in range(3) if other != axis)
        positions = np.flatnonzero(foreground.any(axis=other_axes))
        start = int(positions[0]) // step * step
        stop = (int(positions[-1]) // step + 1) * step
        slices.append(slice(start, stop))
    return labels[tuple(slices)]


def majority_labels(labels, size):
    x, y, z = labels.shape
    blocks = labels.reshape(x // size, size, y // size, size, z)
    counts = np.stack([(blocks == value).sum(axis=(1, 3)) for value in LABELS])
    # LABELS 已按数值排序，票数相同时选较小的原始标签。
    winners = LABELS[counts.argmax(axis=0)]
    return winners.repeat(size, axis=0).repeat(size, axis=1)


def label_statistics(labels):
    labels = analysis_roi(labels)
    block_labels = {size: majority_labels(labels, size) for size in BLOCK_SIZES}
    row = {}
    for region, values in REGIONS.items():
        mask = np.isin(labels, values)
        target = int(mask.sum())
        interior = binary_erosion(mask, structure=AXIAL_NEIGHBORHOOD)
        row[f"{region}_target_voxels"] = target
        row[f"{region}_boundary_voxels"] = target - int(interior.sum())
        for size, approximated in block_labels.items():
            predicted = np.isin(approximated, values)
            denominator = target + int(predicted.sum())
            overlap = int(np.count_nonzero(predicted & mask))
            row[f"{region}_block_{size}_dice"] = 2 * overlap / denominator if denominator else 1.0
    return row


def summarize_labels(rows):
    result = {}
    for region in REGIONS:
        voxels = sum(row[f"{region}_target_voxels"] for row in rows)
        boundary = sum(row[f"{region}_boundary_voxels"] for row in rows)
        result[region] = {
            "pooled_boundary_percent": 100 * boundary / voxels,
            "mean_patient_block_dice_percent": {
                size: 100 * float(np.mean([row[f"{region}_block_{size}_dice"] for row in rows]))
                for size in BLOCK_SIZES
            },
        }
    return result


def read_experiment(directory, expected_ids):
    csv_path = directory / "val_per_patient.csv"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {row["case_id"]: row for row in rows}
    if len(patients) != len(rows) or patients.keys() != expected_ids:
        raise ValueError(f"{directory.name} 的病例与固定 val 清单不匹配")
    scores = np.array([float(row["ET_dice"]) for row in rows])
    if not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError(f"{directory.name} 含无效 ET Dice")
    if summary["selection"] != "best" or not np.isclose(
        scores.mean(), summary["val"]["ET"]["dice"], atol=1e-12, rtol=0
    ):
        raise ValueError(f"{directory.name} 的 val best 摘要与 CSV 不匹配")
    return patients, {
        "experiment": directory.name,
        "val_best_epoch": summary["selected_epoch"],
        "val_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }


def pair_val_patients(label_rows, experiments):
    paired = []
    for label in label_rows:
        case_id = label["case_id"]
        count = label["ET_target_voxels"]
        scores = {}
        for mode, patients in experiments.items():
            patient = patients[case_id]
            if any(int(patient[f"{region}_target_voxels"]) != label[f"{region}_target_voxels"]
                   for region in REGIONS):
                raise ValueError(f"{case_id} 的标签体素数与 {mode} 的 val CSV 不一致")
            scores[mode] = 100 * float(patient["ET_dice"])
        if count:
            paired.append({
                "case_id": case_id,
                "et_target_voxels": count,
                "et_boundary_percent": 100 * label["ET_boundary_voxels"] / count,
                **{f"{mode}_et_dice_percent": score for mode, score in scores.items()},
                "concat_delta_pp": scores["concat"] - scores["baseline"],
                "gated_delta_pp": scores["gated"] - scores["baseline"],
            })
    return paired


def adjusted_associations(rows):
    boundary = np.array([row["et_boundary_percent"] for row in rows])
    log_voxels = np.log([row["et_target_voxels"] for row in rows])
    center = float(np.median(log_voxels))
    design = np.column_stack((np.ones(len(rows)), boundary / 10, log_voxels - center))
    outcomes = np.array([[row["concat_delta_pp"], row["gated_delta_pp"]] for row in rows])
    fit, _, rank, _ = np.linalg.lstsq(design, outcomes, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("设计矩阵不满秩，无法估计体素数调整后的关联")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    slopes = []
    for ids in rng.integers(len(rows), size=(BOOTSTRAP_REPEATS, len(rows))):
        sampled_fit, _, rank, _ = np.linalg.lstsq(design[ids], outcomes[ids], rcond=None)
        if rank == design.shape[1]:
            slopes.append(sampled_fit[1])
    if not slopes:
        raise ValueError("患者重采样均无法估计关联")
    intervals = np.percentile(slopes, [2.5, 97.5], axis=0)
    leave_one_out = []
    for index in range(len(rows)):
        keep = np.arange(len(rows)) != index
        subset_fit, _, rank, _ = np.linalg.lstsq(design[keep], outcomes[keep], rcond=None)
        if rank == design.shape[1]:
            leave_one_out.append(subset_fit[1])
    result = {"n_nonempty_et": len(rows), "log_et_voxels_center": center,
              "bootstrap_requested": BOOTSTRAP_REPEATS, "bootstrap_valid": len(slopes),
              "bootstrap_seed": BOOTSTRAP_SEED, "leave_one_out_valid": len(leave_one_out)}
    for index, mode in enumerate(("concat", "gated")):
        result[mode] = {
            "mean_delta_pp": float(outcomes[:, index].mean()),
            "intercept_at_centered_volume_pp": float(fit[0, index]),
            "slope_pp_per_10pp_boundary": float(fit[1, index]),
            "slope_bootstrap_ci95": intervals[:, index].tolist(),
            "slope_pp_per_log_et_voxel_count": float(fit[2, index]),
            "slope_leave_one_out_range": (
                [float(np.min(leave_one_out, axis=0)[index]), float(np.max(leave_one_out, axis=0)[index])]
                if leave_one_out else None
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE / "data/brats2020/split_seed42.json")
    parser.add_argument("--data-root", type=Path, default=SOURCE.parent / "datasets/BraTS2020")
    parser.add_argument("--output", type=Path, default=SOURCE / "results/analysis/detail-motivation")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest["purpose"] != "training":
        raise ValueError("需要有标签的固定划分清单")
    experiment_root = SOURCE.parent / "research" / "experiments"
    val_ids = set(manifest["splits"]["val"])
    experiments = {}
    provenance = {}
    for mode, exp in EXPERIMENTS.items():
        experiments[mode], provenance[mode] = read_experiment(experiment_root / exp, val_ids)

    rows_by_split = {}
    for split in ("train", "val"):
        rows = []
        ids = manifest["splits"][split]
        print(f"Reading {split} labels ({len(ids)} patients)", flush=True)
        for case_id in ids:
            labels, _ = load_label(manifest["cases"][case_id], args.data_root)
            rows.append({"split": split, "case_id": case_id, **label_statistics(labels)})
        rows_by_split[split] = rows

    paired = pair_val_patients(rows_by_split["val"], experiments)
    summary = {
        "idea": "IDEA-001",
        "manifest_sha256": manifest["sha256"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": list(rows_by_split),
        "block_tie_break": "lowest original BraTS label",
        "block_dice_aggregation": "unweighted 3D patient mean, both empty = 1",
        "experiments": provenance,
        "labels": {
            split: {"n_patients": len(rows), "regions": summarize_labels(rows)}
            for split, rows in rows_by_split.items()
        },
        "val_adjusted_associations": adjusted_associations(paired),
    }
    write_table(args.output / "label-per-patient.csv", rows_by_split["train"] + rows_by_split["val"])
    write_table(args.output / "val-paired.csv", paired)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
