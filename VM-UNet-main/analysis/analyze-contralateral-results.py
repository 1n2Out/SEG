"""复核 EXP-018 对 EXP-006 的验证集差异、异常病例与左右镜像假设。"""
import csv
from importlib import import_module
import json
import sys

import numpy as np
from scipy.stats import spearmanr

full = import_module("compare-detail-options")
sys.path.insert(0, str(full.SOURCE))
from brats.data import check_cache, load_manifest, write_json
from brats.dataset import left_right_axis


def mirror_overlap(case_id, expected_voxels, cache, manifest):
    metadata = check_cache(cache, manifest["cases"][case_id])
    wt = np.load(cache / case_id / "label.npy", mmap_mode="r") != 0
    if np.count_nonzero(wt) != expected_voxels:
        raise ValueError(f"WT 体积与逐病例表不同 {case_id}")
    mirrored = np.flip(wt, axis=1 + left_right_axis(metadata))  # Z,X,Y
    return np.count_nonzero(wt & mirrored) / expected_voxels


def paired_summary(values, resamples):
    interval = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
    return {**full.describe(values), "paired_mean_bootstrap_95": interval.tolist()}


def validation_trajectory(experiment, manifest, case_id):
    directory = full.SOURCE / "results" / experiment["summary"]["run_dir"] / "validation"
    trajectory = {}
    history = full.EXPERIMENTS / experiment["summary"]["exp"] / "history.jsonl"
    for line in history.read_text().splitlines():
        record = json.loads(line)
        if record.get("val_mean_dice") is None:
            continue
        epoch = record["epoch"]
        folder = directory / f"epoch_{epoch:03d}"
        summary = full.read_json(folder / "summary.json")
        if (summary["manifest_sha256"] != manifest["sha256"] or summary["split"] != "val"
                or summary["partial_evaluation"] or summary["n_patients"] != len(manifest["splits"]["val"])
                or not np.isclose(summary["mean_dice"], record["val_mean_dice"], rtol=0, atol=1e-12)):
            raise ValueError(f"验证记录不一致 {folder}")
        with (folder / "per_patient.csv").open() as stream:
            row = next(row for row in csv.DictReader(stream) if row["case_id"] == case_id)
        case_hd95 = np.mean([float(row[r + "_hd95_mm"]) for r in full.REGIONS])
        other_hd95 = (summary["n_patients"] * summary["mean_hd95_mm"] - case_hd95) / (summary["n_patients"] - 1)
        trajectory[epoch] = {"dice": summary["mean_dice"], "hd95": summary["mean_hd95_mm"],
                             "remaining_hd95": other_hd95,
                             "influential_case_et_dice": float(row["ET_dice"])}
    return trajectory


def main():
    reference, candidate = (full.load_experiment(name) for name in ("EXP-006", "EXP-018"))
    manifest = load_manifest(full.SOURCE / "data/brats2020/split_seed42.json")
    cache = full.SOURCE / "data/brats2020/cache"
    ids = sorted(manifest["splits"]["val"])
    controls, patients = reference["patients"], candidate["patients"]
    if set(ids) != patients.keys() or patients.keys() != controls.keys():
        raise ValueError("验证集病例编号不一致")
    for case_id in ids:
        for region in full.REGIONS:
            key = region + "_target_voxels"
            if patients[case_id][key] != controls[case_id][key]:
                raise ValueError(f"{case_id} {region} 真值不一致")

    resamples = np.random.default_rng(42).integers(len(ids), size=(20000, len(ids)))
    summary = {}
    deltas = {}
    for field, scale in (("dice", 100), ("hd95_mm", 1)):
        changes = np.array([[float(patients[i][f"{r}_{field}"]) - float(controls[i][f"{r}_{field}"])
                             for r in full.REGIONS] for i in ids]) * scale
        overall = changes.mean(axis=1)
        summary[field] = {r: full.describe(changes[:, j]) for j, r in enumerate(full.REGIONS)}
        summary[field]["mean_over_regions"] = paired_summary(overall, resamples)
        deltas[field] = overall

    hd = deltas["hd95_mm"]
    order = np.argsort(hd)
    influential = ids[order[0]]
    sensitivity = {}
    for k in (1, 4):
        sensitivity[k] = {"case_ids": [ids[j] for j in order[:k]],
                          "net_reduction_fraction": float(hd[order[:k]].sum() / hd.sum()),
                          "remaining_mean_delta_mm": float(hd[order[k:]].mean())}

    et_fields = ("target_voxels", "prediction_voxels", "dice", "hd95_mm", "recall")
    case_et = {name: {field: float(rows[influential]["ET_" + field]) for field in et_fields}
               for name, rows in (("EXP-006", controls), ("EXP-018", patients))}
    overlap = [mirror_overlap(i, int(patients[i]["WT_target_voxels"]), cache, manifest) for i in ids]
    association = {field: float(spearmanr(overlap, values).statistic) for field, values in deltas.items()}
    old, new = (validation_trajectory(exp, manifest, influential) for exp in (reference, candidate))
    if old.keys() != new.keys():
        raise ValueError("验证轮次不同")
    epochs = [{"epoch": e, "mean_dice_delta_pp": 100 * (new[e]["dice"] - old[e]["dice"]),
               "mean_hd95_delta_mm": new[e]["hd95"] - old[e]["hd95"],
               "remaining_mean_hd95_delta_mm": new[e]["remaining_hd95"] - old[e]["remaining_hd95"],
               "influential_case_et_dice": [old[e]["influential_case_et_dice"],
                                            new[e]["influential_case_et_dice"]]} for e in sorted(old)]
    results = {"reference": "EXP-006", "candidate": "EXP-018", "split": "val",
               "bootstrap": {"unit": "paired patient at fixed best checkpoints", "seed": 42, "draws": 20000},
               "paired_deltas": summary, "hd95_sensitivity": sensitivity, "influential_case": influential,
               "influential_case_et": case_et, "wt_mirror_overlap_spearman": association,
               "epochs": epochs, "case_deltas": [
                   {"case_id": i, "mean_dice_delta_pp": deltas["dice"][j], "mean_hd95_delta_mm": hd[j],
                    "wt_mirror_overlap": overlap[j]} for j, i in enumerate(ids)]}
    write_json(full.EXPERIMENTS / "EXP-018" / "validation-audit.json", results)


if __name__ == "__main__":
    main()
