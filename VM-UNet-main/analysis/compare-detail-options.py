"""TASK-005：EXP-011～014 相对 EXP-006 的 val best 配对分析。

仅读取 collect_brats.py 生成的验证集文件；接受完整协议规定的正常早停。
默认产物：results/analysis/task-005/；不读取或评价测试集。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]
EXPERIMENTS = SOURCE.parent / "research" / "experiments"
REGIONS = ("WT", "TC", "ET")
CANDIDATES = ("EXP-011", "EXP-012", "EXP-013", "EXP-014")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment(exp):
    directory = EXPERIMENTS / exp
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "config.json")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line]
    with (directory / "val_per_patient.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    patients = {row["case_id"]: row for row in rows}
    if len(rows) != 55 or len(patients) != len(rows):
        raise ValueError(exp + " 必须包含 55 个不同 val 患者")
    if config["epochs"] != 100 or summary["selection"] != "best" or config.get("warm_start"):
        raise ValueError(exp + " 不是完整协议的预训练初始化 val best")
    if [row["epoch"] for row in history] != list(range(1, summary["epochs_completed"] + 1)):
        raise ValueError(exp + " 训练轨迹有缺失")
    stopped = history[-1]["early_stopping"]["stopped"]
    if history[-1]["epoch"] != 100 and not stopped:
        raise ValueError(exp + " 既未完成 100 轮，也未按协议早停")
    validated = [row for row in history if row.get("val_mean_dice") is not None]
    best = max(validated, key=lambda row: row["val_mean_dice"])
    if best["epoch"] != summary["selected_epoch"]:
        raise ValueError(exp + " 收集轮次不是 val best")
    if not np.isclose(best["val_mean_dice"], summary["val"]["mean_dice"], rtol=0, atol=1e-12):
        raise ValueError(exp + " history 与 summary 不一致")
    for region in REGIONS:
        for metric in ("dice", "hd95_mm"):
            values = np.asarray([float(row[region + "_" + metric]) for row in rows])
            if not np.isfinite(values).all() or (values < 0).any() or (metric == "dice" and (values > 1).any()):
                raise ValueError(exp + " 存在无效指标")
            if not np.isclose(values.mean(), summary["val"][region][metric], rtol=0, atol=1e-12):
                raise ValueError(exp + " CSV 与 summary 不一致")
    for metric, key in (("dice", "mean_dice"), ("hd95_mm", "mean_hd95_mm")):
        mean = np.mean([summary["val"][r][metric] for r in REGIONS])
        if not np.isclose(mean, summary["val"][key], rtol=0, atol=1e-12):
            raise ValueError(exp + " 宏平均不一致")
    empty = [row for row in rows if int(row["ET_target_voxels"]) == 0]
    fp = sorted(row["case_id"] for row in empty if int(row["ET_prediction_voxels"]) > 0)
    if summary["val"]["et_empty_false_positive"] != f"{len(fp)}/{len(empty)}":
        raise ValueError(exp + " 空 ET 假阳性统计不一致")
    nonempty_mean = np.mean([float(row["ET_dice"]) for row in rows if int(row["ET_target_voxels"]) > 0])
    if not np.isclose(nonempty_mean, summary["val"]["et_nonempty_dice"], rtol=0, atol=1e-12):
        raise ValueError(exp + " 非空 ET 均值不一致")
    hashes = {str((directory / name).relative_to(EXPERIMENTS)):
              hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ("summary.json", "val_per_patient.csv", "config.json", "history.jsonl")}
    return {"summary": summary, "config": config, "patients": patients,
            "empty_et_fp_ids": fp, "early_stopped": stopped, "hashes": hashes}


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "up": int((array > 1e-12).sum()), "down": int((array < -1e-12).sum()),
            "tie": int((np.abs(array) <= 1e-12).sum())}


def compare(exp, baseline, candidate, output):
    expected_config = {k: v for k, v in baseline["config"].items() if k != "detail_fusion"}
    actual_config = {k: v for k, v in candidate["config"].items() if k != "detail_fusion"}
    if actual_config != expected_config:
        raise ValueError(exp + " 除 detail_fusion 外协议配置不同")
    if baseline["patients"].keys() != candidate["patients"].keys():
        raise ValueError(exp + " val 病例集合不同")
    pairs = []
    for case_id in sorted(baseline["patients"]):
        before, after = baseline["patients"][case_id], candidate["patients"][case_id]
        if any(int(before[r + "_target_voxels"]) != int(after[r + "_target_voxels"]) for r in REGIONS):
            raise ValueError(case_id + " 标注体素数不同")
        row = {"case_id": case_id, "et_nonempty": int(before["ET_target_voxels"]) > 0}
        for region in REGIONS:
            for metric, scale, unit in (("dice", 100, "pp"), ("hd95_mm", 1, "mm")):
                row[f"{region}_{metric}_delta_{unit}"] = scale * (float(after[region + "_" + metric]) - float(before[region + "_" + metric]))
        row["mean_dice_delta_pp"] = np.mean([row[r + "_dice_delta_pp"] for r in REGIONS]).item()
        row["mean_hd95_delta_mm"] = np.mean([row[r + "_hd95_mm_delta_mm"] for r in REGIONS]).item()
        pairs.append(row)
    statistics = {key: describe([row[key] for row in pairs])
                  for key in pairs[0] if key.endswith(("_pp", "_mm"))}
    statistics["nonempty_et_dice_delta_pp"] = describe([row["ET_dice_delta_pp"] for row in pairs if row["et_nonempty"]])
    statistics["empty_et_dice_delta_pp"] = describe([row["ET_dice_delta_pp"] for row in pairs if not row["et_nonempty"]])
    contributions = {"empty_et": sum(row["ET_dice_delta_pp"] for row in pairs if not row["et_nonempty"]) / len(pairs) / 3,
                     "nonempty_et": sum(row["ET_dice_delta_pp"] for row in pairs if row["et_nonempty"]) / len(pairs) / 3,
                     "WT": sum(row["WT_dice_delta_pp"] for row in pairs) / len(pairs) / 3,
                     "TC": sum(row["TC_dice_delta_pp"] for row in pairs) / len(pairs) / 3}
    a, b = baseline["summary"]["val"], candidate["summary"]["val"]
    delta = 100 * (b["mean_dice"] - a["mean_dice"])
    nonempty = statistics["nonempty_et_dice_delta_pp"]
    criteria = {"mean_dice_gain_at_least_0_3pp": delta >= 0.3,
                "et_dice_not_lower": b["ET"]["dice"] >= a["ET"]["dice"],
                "nonempty_et_more_up_than_down": nonempty["up"] > nonempty["down"],
                "empty_et_fp_not_increased": len(candidate["empty_et_fp_ids"]) <= len(baseline["empty_et_fp_ids"])}
    result = {"reference": "EXP-006", "candidate": exp, "split": "val", "selection": "best",
              "selected_epoch": candidate["summary"]["selected_epoch"],
              "epochs_completed": candidate["summary"]["epochs_completed"],
              "early_stopped": candidate["early_stopped"],
              "delta_mean_dice_pp": delta, "delta_et_dice_pp": 100 * (b["ET"]["dice"] - a["ET"]["dice"]),
              "delta_mean_hd95_mm": b["mean_hd95_mm"] - a["mean_hd95_mm"],
              "reference_empty_et_fp_ids": baseline["empty_et_fp_ids"],
              "candidate_empty_et_fp_ids": candidate["empty_et_fp_ids"],
              "contributions_to_mean_dice_delta_pp": contributions,
              "paired": statistics, "criteria": criteria,
              "candidate_eligible": all(criteria.values()) if exp != "EXP-014" else None,
              "input_sha256": {**baseline["hashes"], **candidate["hashes"]},
              "largest_losses": sorted(pairs, key=lambda row: row["mean_dice_delta_pp"])[:5],
              "largest_gains": sorted(pairs, key=lambda row: row["mean_dice_delta_pp"], reverse=True)[:5]}
    directory = output / exp.lower()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "val-paired.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    (directory / "comparison-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "results/analysis/task-005")
    args = parser.parse_args()
    baseline = load_experiment("EXP-006")
    results = {exp: compare(exp, baseline, load_experiment(exp), args.output) for exp in CANDIDATES}
    eligible = [exp for exp in CANDIDATES[:3] if results[exp]["candidate_eligible"]]
    recommendation = max(eligible, key=lambda exp: results[exp]["delta_mean_dice_pp"]) if eligible else "EXP-006"
    gate_gain = -results["EXP-014"]["delta_mean_dice_pp"]
    gate_result = "gated_at_least_0_3pp_better" if gate_gain >= 0.3 else (
        "concat_at_least_0_3pp_better" if gate_gain <= -0.3 else "within_0_3pp")
    summary = {"task": "TASK-005", "split": "val", "reference": "EXP-006",
               "eligible_candidates": eligible, "recommended_method": recommendation,
               "gated_minus_concat_mean_dice_pp": gate_gain, "gate_result": gate_result,
               "comparisons": results, "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "comparison-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
