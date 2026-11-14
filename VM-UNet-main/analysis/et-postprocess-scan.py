"""IDEA-004：ET 小体积后处理的阈值扫描，只用 val。

输入：research/experiments/<EXP>/ 下的 val_per_patient.csv 和 summary.json（默认 EXP-001、EXP-006）。
划分：只读 val；本脚本没有读取 test 的选项。
规则：某患者预测的 ET 体素数满足 0 < n < T 时，把这些 ET 全部改判为坏死（标签 1）。
      WT、TC 不变；ET 变空后，真值也为空记 Dice 1、HD95 0，否则记 Dice 0、HD95 为体积对角线。
校验：T=0 必须复现 summary.json 的 mean_dice 和 mean_hd95_mm；单侧为空时的 HD95 必须等于对角线。
产物：results/analysis/et-postprocess/scan.csv（每个 EXP 与阈值一行），并在终端打印同样的表。
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
EXPERIMENTS = SOURCE.parent / "research" / "experiments"
OUTPUT = SOURCE / "results" / "analysis" / "et-postprocess"
REGIONS = ("WT", "TC", "ET")
# BraTS2020 volumes are 240 x 240 x 155 voxels at 1 mm (see metrics.binary_metrics).
DIAGONAL_MM = math.sqrt(240 ** 2 + 240 ** 2 + 155 ** 2)
THRESHOLDS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000)


def read_val(exp):
    directory = EXPERIMENTS / exp
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "val_per_patient.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["case_id"] for row in rows}) != len(rows) or not rows:
        raise ValueError(f"{exp}: val_per_patient.csv 患者重复或为空")
    for row in rows:
        for region in REGIONS:
            target, predicted = int(row[f"{region}_target_voxels"]), int(row[f"{region}_prediction_voxels"])
            if (target == 0) != (predicted == 0) and abs(float(row[f"{region}_hd95_mm"]) - DIAGONAL_MM) > 1e-6:
                raise ValueError(f"{exp}/{row['case_id']}: 单侧为空的 HD95 与体积对角线不符")
    return summary["val"], rows


def apply_threshold(rows, threshold):
    """Per-patient (WT, TC, ET) Dice and HD95 after the rule, plus counts of changed patients."""
    dice, hd95 = [], []
    fixed = harmed = false_positive = empty = 0
    for row in rows:
        target, predicted = int(row["ET_target_voxels"]), int(row["ET_prediction_voxels"])
        et_dice, et_hd95 = float(row["ET_dice"]), float(row["ET_hd95_mm"])
        if 0 < predicted < threshold:
            predicted = 0
            if target == 0:
                fixed += 1
                et_dice, et_hd95 = 1.0, 0.0
            else:
                harmed += 1
                et_dice, et_hd95 = 0.0, DIAGONAL_MM
        if target == 0:
            empty += 1
            false_positive += predicted > 0
        dice.append((float(row["WT_dice"]), float(row["TC_dice"]), et_dice))
        hd95.append((float(row["WT_hd95_mm"]), float(row["TC_hd95_mm"]), et_hd95))
    regional_dice = [sum(values) / len(rows) for values in zip(*dice)]
    regional_hd95 = [sum(values) / len(rows) for values in zip(*hd95)]
    return {"mean_dice": sum(regional_dice) / 3, "mean_hd95_mm": sum(regional_hd95) / 3,
            "et_dice": regional_dice[2], "et_hd95_mm": regional_hd95[2],
            "et_empty_false_positive": f"{false_positive}/{empty}",
            "cleared_empty_patients": fixed, "cleared_nonempty_patients": harmed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exps", nargs="+", default=["EXP-001", "EXP-006"])
    parser.add_argument("--thresholds", nargs="+", type=int, default=list(THRESHOLDS))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Chinese table headers on Windows consoles
    if 0 not in args.thresholds or any(t < 0 for t in args.thresholds):
        raise SystemExit("阈值须为非负整数，且包含 0 作为原始输出对照")
    records = []
    for exp in args.exps:
        summary, rows = read_val(exp)
        original = apply_threshold(rows, 0)
        if (abs(original["mean_dice"] - summary["mean_dice"]) > 1e-9
                or abs(original["mean_hd95_mm"] - summary["mean_hd95_mm"]) > 1e-9):
            raise ValueError(f"{exp}: T=0 未能复现 summary.json")
        for threshold in sorted(set(args.thresholds)):
            result = apply_threshold(rows, threshold)
            result["delta_mean_dice"] = result["mean_dice"] - original["mean_dice"]
            records.append({"exp": exp, "threshold": threshold, **result})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("| EXP | T | Mean Dice | ΔMean | ET Dice | Mean HD95 | 空ET假阳性 | 修正空ET | 误删非空ET |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        print(f"| {r['exp']} | {r['threshold']} | {100 * r['mean_dice']:.2f} | {100 * r['delta_mean_dice']:+.2f} | "
              f"{100 * r['et_dice']:.2f} | {r['mean_hd95_mm']:.2f} | {r['et_empty_false_positive']} | "
              f"{r['cleared_empty_patients']} | {r['cleared_nonempty_patients']} |")
    print("产物：", OUTPUT / "scan.csv")


if __name__ == "__main__":
    main()
