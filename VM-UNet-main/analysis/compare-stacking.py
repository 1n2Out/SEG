"""TASK-004：核验固定第 10 轮 val 结果，逐患者配对并执行预定筛选决策。

输入为 EXP-007～010 的 summary.json、config.json、val_per_patient.csv。
复用 compare-screening.py 的指标核验和配对统计，不读取 test 或 final/。
产物写入 results/analysis/task-004/；重复执行只覆盖自身产物。
"""
import importlib.util
import json
from pathlib import Path


HELPER = Path(__file__).with_name("compare-screening.py")
SPEC = importlib.util.spec_from_file_location("screening_comparison", HELPER)
SCREENING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREENING)
OUTPUT = SCREENING.SOURCE / "results" / "analysis" / "task-004"
IDS = ("EXP-007", "EXP-008", "EXP-009", "EXP-010")


def common_config(config):
    result = {key: value for key, value in config.items() if key != "detail_fusion"}
    result["warm_start"] = {key: value for key, value in config["warm_start"].items()
                            if key != "new_parameters"}
    return result


def main():
    experiments = {exp: SCREENING.read_experiment(exp) for exp in IDS}
    reference = experiments["EXP-007"]
    expected = {
        "EXP-007": (0.0, 0, []),
        "EXP-008": (0.5, 0, ["detail_fusion.boundary_head."]),
        "EXP-009": (0.0, 1, ["detail_fusion.context."]),
        "EXP-010": (0.5, 1, ["detail_fusion.boundary_head.", "detail_fusion.context."]),
    }
    for exp, record in experiments.items():
        config = record["config"]
        if common_config(config) != common_config(reference["config"]):
            raise ValueError(f"{exp} 的公共配置或热启动来源与对照不同")
        git = record["summary"]["git"]
        if git["commit"] != reference["summary"]["git"]["commit"] or git["dirty"]:
            raise ValueError(f"{exp} 的来源提交不同或训练时源码有未提交改动")
        weight, slices, prefixes = expected[exp]
        detail = {**reference["config"]["detail_fusion"],
                  "boundary_weight": weight, "context_slices": slices}
        if config["detail_fusion"] != detail or config["warm_start"]["new_parameters"] != prefixes:
            raise ValueError(f"{exp} 的叠加配置不符合任务单")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    SCREENING.OUTPUT = OUTPUT
    comparisons = {exp: SCREENING.compare(exp, "EXP-007", experiments) for exp in IDS[1:]}
    screening = {}
    for exp, comparison in comparisons.items():
        criteria = {
            "mean_dice_gain_at_least_0_5_pp": comparison["mean_dice_delta_pp"]["mean"] >= 0.5,
            "et_dice_increased": comparison["ET_dice_delta_pp"]["mean"] > 0,
            "mean_hd95_not_worse": comparison["mean_hd95_delta_mm"]["mean"] <= 0,
            "empty_et_false_positives_not_increased":
                len(comparison["candidate_empty_et_false_positive_ids"])
                <= len(comparison["reference_empty_et_false_positive_ids"]),
        }
        screening[exp] = {"criteria": criteria, "passed": all(criteria.values())}
    if screening["EXP-010"]["passed"]:
        full_training = ["gated-boundary-context", "gated-boundary", "gated-context"]
    else:
        full_training = [name for exp, name in (("EXP-008", "gated-boundary"),
                                              ("EXP-009", "gated-context"))
                         if screening[exp]["passed"]]
    result = {
        "task": "TASK-004", "split": "val", "selected_epoch": 10, "reference": "EXP-007",
        "difference_direction": "candidate minus reference; Dice: percentage points; HD95: mm",
        "input_sha256": {exp: record["input_sha256"] for exp, record in experiments.items()},
        "comparisons": comparisons, "screening": screening, "full_training": full_training,
    }
    target = OUTPUT / "comparison-summary.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("分析产物：", target)
    print(json.dumps({"screening": screening, "full_training": full_training,
                      "comparisons": {exp: {key: value[key] for key in
                        ("mean_dice_delta_pp", "ET_dice_delta_pp", "mean_hd95_delta_mm",
                         "nonempty_et_dice_delta_pp", "candidate_empty_et_false_positive_ids")}
                        for exp, value in comparisons.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
