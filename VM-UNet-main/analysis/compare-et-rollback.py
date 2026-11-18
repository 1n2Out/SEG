"""TASK-007：读取 EXP-006/015/016 的验证集摘要和逐患者 CSV。

两组候选固定第 10 轮。结果默认写入 results/analysis/task-007/summary.json。
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from brats.data import write_json

REGIONS = ("WT", "TC", "ET")


def read_experiment(root, name, epoch, mode=None):
    directory = root / name
    summary_path = directory / "summary.json"
    csv_path = directory / "val_per_patient.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["exp"] != name or summary["selected_epoch"] != epoch:
        raise ValueError(f"{name} 的实验编号或轮次不匹配")
    if mode and (summary["epochs_completed"] != epoch or summary["selection"] != "fixed" or
                 summary["config"]["extension"]["detail_fusion"]["et_rollback"] != mode or
                 Path(summary["config"]["warm_start"]).parent.name != "brats2020_detail_gated_1113_1530"):
        raise ValueError(f"{name} 的筛选配置或 EXP-006 来源不匹配")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {r["case_id"]: r for r in rows}
    if len(rows) != 55 or len(patients) != len(rows):
        raise ValueError(f"{name} 应包含 55 个不同的验证病例")
    for region in REGIONS:
        scores = [float(row[f"{region}_dice"]) for row in rows]
        if not all(0 <= value <= 1 for value in scores) or not math.isclose(
                mean(scores), summary["val"][region]["dice"], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"{name} 的 {region} Dice 与摘要不一致")
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (summary_path, csv_path)}
    return summary["val"], patients, files


def et_counts(row):
    target = int(row["ET_target_voxels"])
    predicted = int(row["ET_prediction_voxels"])
    intersection = float(row["ET_dice"]) * (target + predicted) / 2
    tp = round(intersection)
    if abs(tp - intersection) > 1e-6 or not 0 <= tp <= min(target, predicted):
        raise ValueError(f"{row['case_id']} 的 ET 体素数与 Dice 不一致")
    return tp, predicted - tp


def compare(reference, candidate):
    ref_summary, ref_patients, _ = reference
    val, patients, _ = candidate
    if patients.keys() != ref_patients.keys():
        raise ValueError("验证病例集合不一致")
    differences, caught, false_positives, newly_false_empty = [], 0, 0, []
    for case_id, ref in ref_patients.items():
        row = patients[case_id]
        if any(row[f"{region}_target_voxels"] != ref[f"{region}_target_voxels"] for region in REGIONS):
            raise ValueError(f"{case_id} 的标注体素数不一致")
        old_tp, old_fp = et_counts(ref)
        new_tp, new_fp = et_counts(row)
        if new_tp < old_tp or new_fp < old_fp:
            raise ValueError(f"{case_id} 的 ET 真阳性或假阳性减少，须复核冻结模型和原验证环境")
        caught += new_tp - old_tp
        false_positives += new_fp - old_fp
        if int(ref["ET_target_voxels"]) > 0:
            differences.append(100 * (float(row["ET_dice"]) - float(ref["ET_dice"])))
        elif old_fp == 0 and new_fp > 0:
            newly_false_empty.append(case_id)
    up = sum(value > 1e-10 for value in differences)
    down = sum(value < -1e-10 for value in differences)
    delta_mean = 100 * (val["mean_dice"] - ref_summary["mean_dice"])
    delta_et = 100 * (val["ET"]["dice"] - ref_summary["ET"]["dice"])
    checks = {
        "mean_dice_at_least_0_3pp": delta_mean >= 0.3 - 1e-10,
        "et_dice_not_lower": delta_et >= -1e-10,
        "nonempty_more_gains_than_losses": up > down,
        "no_new_empty_et_false_positive_patient": not newly_false_empty,
    }
    return {
        "mean_dice_delta_pp": delta_mean,
        "et_dice_delta_pp": delta_et,
        "mean_hd95_delta_mm": val["mean_hd95_mm"] - ref_summary["mean_hd95_mm"],
        "nonempty_et": {"n": len(differences), "mean_delta_pp": mean(differences),
                        "median_delta_pp": median(differences), "up": up, "down": down,
                        "tie": len(differences) - up - down},
        "recovered_true_et_voxels": caught,
        "added_false_et_voxels": false_positives,
        "new_empty_et_false_positive_cases": newly_false_empty,
        "checks": checks,
        "passes_vs_exp006": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=SOURCE.parent / "research/experiments")
    parser.add_argument("--output", type=Path, default=SOURCE / "results/analysis/task-007/summary.json")
    args = parser.parse_args()
    reference = read_experiment(args.experiments, "EXP-006", 95)
    constant = read_experiment(args.experiments, "EXP-015", 10, "constant")
    local = read_experiment(args.experiments, "EXP-016", 10, "local")
    results = {"constant": compare(reference, constant), "local": compare(reference, local)}
    local_delta = 100 * (local[0]["mean_dice"] - constant[0]["mean_dice"])
    result = {"split": "val", "reference": "EXP-006", "constant": "EXP-015", "local": "EXP-016",
              "comparisons": results, "local_vs_constant_mean_dice_pp": local_delta,
              "local_candidate_for_full_training": results["local"]["passes_vs_exp006"] and local_delta > 1e-10,
              "input_sha256": {**reference[2], **constant[2], **local[2]}}
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
