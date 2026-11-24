"""比较候选与 EXP-006、EXP-001 的 val best。默认复核 TASK-008。"""
import argparse
from importlib import import_module
import json
from statistics import mean


# 复用完整训练的轮次、CSV、汇总指标和空 ET 假阳性校验。
full_training = import_module("compare-detail-options")


def compare(candidate, reference):
    patients = candidate["patients"]
    controls = reference["patients"]
    if patients.keys() != controls.keys():
        raise ValueError("验证病例集合不同")
    groups = {True: [], False: []}
    patient_deltas = {}
    for case_id, row in patients.items():
        control = controls[case_id]
        differences = []
        for region in full_training.REGIONS:
            key = region + "_target_voxels"
            if int(row[key]) != int(control[key]):
                raise ValueError(f"{case_id} 的 {region} 真值体素数不同")
            key = region + "_dice"
            differences.append(100 * (float(row[key]) - float(control[key])))
        patient_deltas[case_id] = mean(differences)
        groups[int(row["ET_target_voxels"]) > 0].append(patient_deltas[case_id])

    new, old = candidate["summary"]["val"], reference["summary"]["val"]
    new_fp, old_fp = set(candidate["empty_et_fp_ids"]), set(reference["empty_et_fp_ids"])
    worst = sorted(patient_deltas, key=patient_deltas.get)[:3]
    nonempty_region_errors = {}
    for region in full_training.REGIONS:
        ids = [i for i in patients if int(patients[i][region + "_target_voxels"]) > 0]
        nonempty_region_errors[region] = {"n": len(ids)}
        for metric in ("precision", "recall"):
            key = region + "_" + metric
            nonempty_region_errors[region][metric + "_delta_pp"] = mean(
                100 * (float(patients[i][key]) - float(controls[i][key])) for i in ids)
    return {
        "n": len(patients),
        "mean_dice_delta_pp": 100 * (new["mean_dice"] - old["mean_dice"]),
        "region_dice_delta_pp": {
            r: 100 * (new[r]["dice"] - old[r]["dice"]) for r in full_training.REGIONS
        },
        "mean_hd95_delta_mm": new["mean_hd95_mm"] - old["mean_hd95_mm"],
        "nonempty_et_paired_mean_dice_delta_pp": full_training.describe(groups[True]),
        "empty_et_paired_mean_dice_delta_pp": full_training.describe(groups[False]),
        "empty_et_fp_added": sorted(new_fp - old_fp),
        "empty_et_fp_resolved": sorted(old_fp - new_fp),
        "empty_et_fp_persistent": sorted(new_fp & old_fp),
        "nonempty_region_errors": nonempty_region_errors,
        "worst_three_mean_dice_delta_pp": {i: patient_deltas[i] for i in worst},
        "worst_three_contribution_to_full_mean_pp": sum(patient_deltas[i] for i in worst) / len(patients),
        "remaining_patients_mean_delta_pp": mean(d for i, d in patient_deltas.items() if i not in worst),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("EXP-017", "EXP-018"), default="EXP-017")
    experiment = parser.parse_args().candidate
    candidate = full_training.load_experiment(experiment)
    results = {name: compare(candidate, full_training.load_experiment(name))
               for name in ("EXP-006", "EXP-001")}
    path = full_training.EXPERIMENTS / experiment / "comparison-summary.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
