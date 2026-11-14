"""EXP-006 完整训练与 EXP-001 的配对比较。

输入：两实验的 summary.json、history.jsonl、val_per_patient.csv；仅分析 val。
TASK-003 FINAL 授权后，显式 --include-test 才读取 final/test_summary.json 和
final/test_per_patient.csv。测试仅作固定模型的最终描述，不用于选模型或调参。
输出：results/analysis/exp-006/ 下的 val CSV/JSON；测试产物单独写入 final/。
所有源结果只读；bootstrap 为固定模型的患者抽样区间，不表示训练种子稳定性。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]
EXPERIMENTS = SOURCE.parent / "research" / "experiments"
OUTPUT = SOURCE / "results" / "analysis" / "exp-006"
REGIONS = ("WT", "TC", "ET")
METRICS = ("dice", "hd95_mm", "precision", "recall", "specificity")


def read_json(path):
    return json.loads(path.read_text())


def read_results(exp, split):
    directory = EXPERIMENTS / exp
    summary_path = directory / ("summary.json" if split == "val" else "final/test_summary.json")
    csv_path = directory / ("val_per_patient.csv" if split == "val" else "final/test_per_patient.csv")
    summary = read_json(summary_path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {row["case_id"]: row for row in rows}
    if len(rows) != 55 or len(patients) != len(rows):
        raise ValueError(f"{exp}/{split} 必须包含 55 个不同患者")
    if split == "val":
        if summary["selection"] != "best" or summary["epochs_completed"] != 100:
            raise ValueError("完整训练比较必须使用 100 轮协议的 val best")
        expected = summary["val"]
    else:
        if summary["split"] != "test" or summary["partial_evaluation"] or summary["n_patients"] != 55:
            raise ValueError("FINAL 必须是完整 test 评价")
        if summary["checkpoint_epoch"] != read_json(directory / "summary.json")["selected_epoch"]:
            raise ValueError("测试权重与 val best 不一致")
        expected = {region: {metric: summary["regions"][region][metric]["mean"]
                             for metric in METRICS} for region in REGIONS}
        expected.update({key: summary[key] for key in ("mean_dice", "mean_hd95_mm")})
    aggregate = {}
    for region in REGIONS:
        aggregate[region] = {}
        for metric in METRICS:
            values = np.asarray([float(row[f"{region}_{metric}"]) for row in rows])
            if not np.isfinite(values).all() or (values < 0).any() or (metric != "hd95_mm" and (values > 1).any()):
                raise ValueError("指标超出有效范围")
            aggregate[region][metric] = float(values.mean())
            if metric in expected[region] and not np.isclose(values.mean(), expected[region][metric], rtol=0, atol=1e-12):
                raise ValueError(f"{exp}/{split} CSV 与 summary 不一致")
    aggregate["mean_dice"] = float(np.mean([aggregate[r]["dice"] for r in REGIONS]))
    aggregate["mean_hd95_mm"] = float(np.mean([aggregate[r]["hd95_mm"] for r in REGIONS]))
    for metric in ("mean_dice", "mean_hd95_mm"):
        if not np.isclose(aggregate[metric], expected[metric], rtol=0, atol=1e-12):
            raise ValueError("宏平均不一致")
    empty = [r for r in rows if int(r["ET_target_voxels"]) == 0]
    aggregate["empty_et_n"] = len(empty)
    aggregate["empty_et_false_positive_ids"] = [r["case_id"] for r in empty if int(r["ET_prediction_voxels"]) > 0]
    nonempty = [float(r["ET_dice"]) for r in rows if int(r["ET_target_voxels"]) > 0]
    aggregate["nonempty_et_dice"] = float(np.mean(nonempty))
    hashes = {str(path.relative_to(EXPERIMENTS)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (summary_path, csv_path)}
    return patients, aggregate, hashes, summary


def describe(values, bootstrap=False):
    array = np.asarray(values, dtype=float)
    result = {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
              "positive": int((array > 1e-12).sum()), "negative": int((array < -1e-12).sum()),
              "tie": int((np.abs(array) <= 1e-12).sum())}
    if bootstrap:
        samples = np.random.default_rng(42).integers(0, len(array), size=(10000, len(array)))
        interval = np.percentile(array[samples].mean(axis=1), [2.5, 97.5])
        result["bootstrap_95_ci"] = [float(x) for x in interval]
    return result


def compare(split):
    reference, baseline, hashes_a, summary_a = read_results("EXP-001", split)
    candidate, improved, hashes_b, summary_b = read_results("EXP-006", split)
    if candidate.keys() != reference.keys():
        raise ValueError("病例集合不同")
    if split == "test" and summary_a["manifest_sha256"] != summary_b["manifest_sha256"]:
        raise ValueError("测试划分 SHA256 不同")
    pairs = []
    for case_id in sorted(candidate):
        a, b = candidate[case_id], reference[case_id]
        if any(int(a[f"{r}_target_voxels"]) != int(b[f"{r}_target_voxels"]) for r in REGIONS):
            raise ValueError(f"标签体素数不同：{case_id}")
        row = {"case_id": case_id, "et_nonempty": int(a["ET_target_voxels"]) > 0}
        for region in REGIONS:
            row[f"{region}_dice_delta_pp"] = 100 * (float(a[f"{region}_dice"]) - float(b[f"{region}_dice"]))
            row[f"{region}_hd95_delta_mm"] = float(a[f"{region}_hd95_mm"]) - float(b[f"{region}_hd95_mm"])
        row["mean_dice_delta_pp"] = sum(row[f"{r}_dice_delta_pp"] for r in REGIONS) / 3
        row["mean_hd95_delta_mm"] = sum(row[f"{r}_hd95_delta_mm"] for r in REGIONS) / 3
        pairs.append(row)
    keys = [key for key in pairs[0] if key.endswith(("_pp", "_mm"))]
    statistics = {key: describe([row[key] for row in pairs], key == "mean_dice_delta_pp") for key in keys}
    statistics["nonempty_et_dice_delta_pp"] = describe([row["ET_dice_delta_pp"] for row in pairs if row["et_nonempty"]], True)
    criteria = {"mean_dice_increased": improved["mean_dice"] > baseline["mean_dice"],
                "et_dice_not_lower": improved["ET"]["dice"] >= baseline["ET"]["dice"],
                "mean_hd95_not_worse": improved["mean_hd95_mm"] <= baseline["mean_hd95_mm"],
                "empty_et_fp_not_increased": len(improved["empty_et_false_positive_ids"])
                <= len(baseline["empty_et_false_positive_ids"])}
    result = {"split": split, "reference": "EXP-001", "candidate": "EXP-006",
              "baseline": baseline, "improved": improved, "paired": statistics,
              "input_sha256": {**hashes_a, **hashes_b}, "criteria": criteria,
              "all_criteria_met": all(criteria.values()),
              "bootstrap": {"samples": 10000, "seed": 42, "scope": "paired patient resampling, fixed trained models"}}
    if split == "val":
        trajectories = {}
        for exp in ("EXP-001", "EXP-006"):
            rows = [json.loads(line) for line in (EXPERIMENTS / exp / "history.jsonl").read_text().splitlines() if line.strip()]
            validated = [r for r in rows if r.get("val_mean_dice") is not None]
            early_best = max((r for r in validated if r["epoch"] <= 60), key=lambda r: r["val_mean_dice"])
            trajectories[exp] = {"best_within_first_60": {"epoch": early_best["epoch"], "mean_dice": early_best["val_mean_dice"]},
                                 "best_epoch": rows[-1]["best_epoch"], "best_mean_dice": rows[-1]["best_val_mean_dice"],
                                 "epochs_completed": rows[-1]["epoch"]}
        result["trajectories"] = trajectories
    ranked = sorted(pairs, key=lambda row: row["mean_dice_delta_pp"])
    result["largest_losses"] = ranked[:5]
    result["largest_gains"] = ranked[-5:][::-1]
    worst_hd95 = sorted(pairs, key=lambda row: row["ET_hd95_delta_mm"], reverse=True)[:5]
    result["largest_et_hd95_worsening"] = [
        {"case_id": row["case_id"], "delta_mm": row["ET_hd95_delta_mm"],
         "baseline_et_hd95_mm": float(reference[row["case_id"]]["ET_hd95_mm"]),
         "improved_et_hd95_mm": float(candidate[row["case_id"]]["ET_hd95_mm"]),
         "target_et_voxels": int(candidate[row["case_id"]]["ET_target_voxels"])}
        for row in worst_hd95]
    result["two_largest_et_hd95_contribution_to_mean_hd95_mm"] = sum(
        row["ET_hd95_delta_mm"] for row in worst_hd95[:2]) / len(pairs) / 3
    directory = OUTPUT if split == "val" else OUTPUT / "final"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{split}-paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    (directory / "comparison-summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("split", "baseline", "improved", "paired", "criteria")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-test", action="store_true", help="仅限用户授权 TASK-003 FINAL 分析")
    args = parser.parse_args()
    compare("val")
    if args.include_test:
        compare("test")


if __name__ == "__main__":
    main()
