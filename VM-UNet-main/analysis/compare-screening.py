"""TASK-001：比较第 10 轮热启动筛选结果，不加载模型。

输入：research/experiments/EXP-002 至 EXP-005 的 summary.json、config.json、
val_per_patient.csv。只分析 val，不读取 test 或 final/。
输出：results/analysis/task-001/ 下的配对 CSV 和 comparison-summary.json。
源结果只读；重新运行会覆盖本脚本自己的分析产物。
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]
EXPERIMENTS = SOURCE.parent / "research" / "experiments"
OUTPUT = SOURCE / "results" / "analysis" / "task-001"
REGIONS = ("WT", "TC", "ET")
IDS = ("EXP-002", "EXP-003", "EXP-004", "EXP-005")


def read_experiment(exp):
    directory = EXPERIMENTS / exp
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if (summary["selected_epoch"] != 10 or summary["selection"] != "fixed"
            or summary["epochs_completed"] != 10):
        raise ValueError(f"{exp} 必须使用完成 10 轮后的固定第 10 轮结果")
    with (directory / "val_per_patient.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    patients = {row["case_id"]: row for row in rows}
    if len(rows) != 55 or len(patients) != len(rows):
        raise ValueError(f"{exp} 必须包含 55 个不同的 val 患者")
    for row in patients.values():
        for region in REGIONS:
            for metric in ("dice", "hd95_mm"):
                value = float(row[f"{region}_{metric}"])
                if not math.isfinite(value) or value < 0 or (metric == "dice" and value > 1):
                    raise ValueError(f"{exp} 指标范围异常")
            for field in ("target_voxels", "prediction_voxels"):
                if int(row[f"{region}_{field}"]) < 0:
                    raise ValueError(f"{exp} 体素数异常")
    for metric in ("dice", "hd95_mm"):
        for region in REGIONS:
            actual = np.mean([float(row[f"{region}_{metric}"]) for row in rows])
            if not np.isclose(actual, summary["val"][region][metric], rtol=0, atol=1e-12):
                raise ValueError(f"{exp} CSV 与 summary 不一致：{region}_{metric}")
        actual = np.mean([summary["val"][r][metric] for r in REGIONS])
        key = "mean_dice" if metric == "dice" else "mean_hd95_mm"
        if not np.isclose(actual, summary["val"][key], rtol=0, atol=1e-12):
            raise ValueError(f"{exp} 宏平均与 summary 不一致")
    empty = [row for row in rows if int(row["ET_target_voxels"]) == 0]
    fp = sum(int(row["ET_prediction_voxels"]) > 0 for row in empty)
    if summary["val"]["et_empty_false_positive"] != f"{fp}/{len(empty)}":
        raise ValueError(f"{exp} 空 ET 统计不一致")
    nonempty = [float(row["ET_dice"]) for row in rows if int(row["ET_target_voxels"]) > 0]
    if not np.isclose(np.mean(nonempty), summary["val"]["et_nonempty_dice"], rtol=0, atol=1e-12):
        raise ValueError(f"{exp} 非空 ET 统计不一致")
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ("summary.json", "config.json", "val_per_patient.csv")}
    return {"summary": summary, "config": config, "patients": patients, "input_sha256": hashes}


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "positive": int((array > 1e-12).sum()), "negative": int((array < -1e-12).sum()),
            "tie": int((np.abs(array) <= 1e-12).sum())}


def compare(candidate, reference, experiments):
    left = experiments[candidate]["patients"]
    right = experiments[reference]["patients"]
    if left.keys() != right.keys():
        raise ValueError("患者集合不同，无法配对比较")
    rows = []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if any(a[f"{r}_target_voxels"] != b[f"{r}_target_voxels"] for r in REGIONS):
            raise ValueError(f"患者标签体素数不同：{case_id}")
        row = {"case_id": case_id, "et_nonempty": int(a["ET_target_voxels"]) > 0}
        for region in REGIONS:
            row[f"{region}_dice_delta_pp"] = 100 * (float(a[f"{region}_dice"]) - float(b[f"{region}_dice"]))
            row[f"{region}_hd95_delta_mm"] = float(a[f"{region}_hd95_mm"]) - float(b[f"{region}_hd95_mm"])
        row["mean_dice_delta_pp"] = sum(row[f"{r}_dice_delta_pp"] for r in REGIONS) / 3
        row["mean_hd95_delta_mm"] = sum(row[f"{r}_hd95_delta_mm"] for r in REGIONS) / 3
        row["candidate_et_false_positive"] = not row["et_nonempty"] and int(a["ET_prediction_voxels"]) > 0
        row["reference_et_false_positive"] = not row["et_nonempty"] and int(b["ET_prediction_voxels"]) > 0
        rows.append(row)
    metrics = [f"{r}_{m}" for r in REGIONS for m in ("dice_delta_pp", "hd95_delta_mm")]
    metrics += ["mean_dice_delta_pp", "mean_hd95_delta_mm"]
    stats = {key: describe([row[key] for row in rows]) for key in metrics}
    stats["nonempty_et_dice_delta_pp"] = describe([row["ET_dice_delta_pp"] for row in rows if row["et_nonempty"]])
    stats["candidate_empty_et_false_positive_ids"] = [r["case_id"] for r in rows if r["candidate_et_false_positive"]]
    stats["reference_empty_et_false_positive_ids"] = [r["case_id"] for r in rows if r["reference_et_false_positive"]]
    ranked = sorted(rows, key=lambda row: row["mean_dice_delta_pp"])
    stats["largest_losses"] = ranked[:5]
    stats["largest_gains"] = ranked[-5:][::-1]
    filename = f"{candidate.lower()}-vs-{reference.lower()}.csv"
    with (OUTPUT / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stats


def main():
    experiments = {exp: read_experiment(exp) for exp in IDS}
    baseline_config = experiments["EXP-002"]["config"]
    for exp, record in experiments.items():
        common = {key: value for key, value in record["config"].items()
                  if key not in ("detail_fusion", "shallow_refinement")}
        if common != baseline_config:
            raise ValueError(f"{exp} 与热启动对照组的公共配置不同")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = (("EXP-003", "EXP-002"), ("EXP-004", "EXP-002"), ("EXP-005", "EXP-002"),
             ("EXP-003", "EXP-004"), ("EXP-003", "EXP-005"))
    comparisons = {f"{a}-vs-{b}": compare(a, b, experiments) for a, b in pairs}
    primary = comparisons["EXP-003-vs-EXP-002"]
    criteria = {
        "mean_dice_gain_at_least_0_5_pp": primary["mean_dice_delta_pp"]["mean"] >= 0.5,
        "et_dice_increased": primary["ET_dice_delta_pp"]["mean"] > 0,
        "mean_hd95_not_worse": primary["mean_hd95_delta_mm"]["mean"] <= 0,
        "empty_et_false_positives_not_increased": len(primary["candidate_empty_et_false_positive_ids"])
        <= len(primary["reference_empty_et_false_positive_ids"]),
    }
    result = {"task": "TASK-001", "split": "val", "selected_epoch": 10,
              "difference_direction": "candidate minus reference; Dice: percentage points; HD95: mm",
              "input_sha256": {exp: record["input_sha256"] for exp, record in experiments.items()},
              "comparisons": comparisons, "criteria": criteria, "screening_passed": all(criteria.values())}
    target = OUTPUT / "comparison-summary.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("分析产物：", target)
    print(json.dumps({"criteria": criteria, "screening_passed": result["screening_passed"],
                      "comparisons": {key: {metric: value[metric] for metric in
                        ("mean_dice_delta_pp", "ET_dice_delta_pp", "mean_hd95_delta_mm", "nonempty_et_dice_delta_pp")}
                        for key, value in comparisons.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
