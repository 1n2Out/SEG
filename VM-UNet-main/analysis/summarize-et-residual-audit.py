"""TASK-006：离线核验三张分析表，汇总体素修正与原始标签转移；不运行推理。"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
ACTIONS = ("recover", "remove-fp", "lose", "add-fp")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    patients = read_csv(directory / "patients.csv")
    regions = read_csv(directory / "regions.csv")
    transitions = read_csv(directory / "transitions.csv")
    archive = SOURCE.parent / "research/experiments/EXP-006/val_per_patient.csv"
    references = {r["case_id"]: r for r in read_csv(archive)}
    if len(patients) != 55 or {r["case_id"] for r in patients} != references.keys():
        raise ValueError("Expected the same 55 unique validation patients")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != summary["provenance"]["reference_sha256"]:
        raise ValueError("Archived reference changed")
    by_region, by_transition = {}, Counter()
    for row in regions:
        key = (row["case_id"], row["zone"], row["action"])
        if key in by_region:
            raise ValueError("Duplicate regional row")
        count, opportunities = int(row["count"]), int(row["opportunities"])
        assert 0 <= count <= opportunities
        assert (row["rate"] == "") == (opportunities == 0)
        if opportunities:
            assert np.isclose(float(row["rate"]), count / opportunities, atol=1e-12, rtol=0)
        by_region[key] = count
    for row in transitions:
        assert (int(row["trunk"]) == 4) != (int(row["final"]) == 4)
        expected = ("recover" if int(row["final"]) == 4 else "lose") if int(row["truth"]) == 4 else (
            "add-fp" if int(row["final"]) == 4 else "remove-fp")
        assert row["action"] == expected
        key = (row["case_id"], row["zone"], row["action"])
        assert key in by_region
        by_transition[key] += int(row["count"])
    assert all(by_transition[key] == count for key, count in by_region.items())
    for row in patients:
        case_id = row["case_id"]
        reference = references[case_id]
        assert int(row["target_et"]) == int(reference["ET_target_voxels"])
        assert int(row["final_et"]) == int(reference["ET_prediction_voxels"])
        assert np.isclose(float(row["final_dice"]), float(reference["ET_dice"]), atol=1e-12, rtol=0)
        for action in ACTIONS:
            assert int(row[action]) == sum(count for (case, _, a), count in by_region.items()
                                           if case == case_id and a == action)
        assert int(row["final_et"]) == (int(row["trunk_et"]) + int(row["recover"]) + int(row["add-fp"])
                                        - int(row["remove-fp"]) - int(row["lose"]))
    groups = {}
    for name, rows in (("all", patients), ("nonempty", [r for r in patients if int(r["target_et"])]),
                       ("empty", [r for r in patients if not int(r["target_et"])])):
        ids = {r["case_id"] for r in rows}
        delta = np.array([float(r["delta_pp"]) for r in rows])
        paired = dict(n=len(rows), mean_pp=float(delta.mean()), median_pp=float(np.median(delta)),
                      gains=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
        assert all(np.isclose(value, summary["paired"][name][key], atol=1e-12, rtol=0)
                   for key, value in paired.items())
        pooled_regions, label_changes = {}, Counter()
        for row in regions:
            if row["case_id"] in ids:
                target = pooled_regions.setdefault(row["zone"], {})
                target[row["action"]] = target.get(row["action"], 0) + int(row["count"])
        for row in transitions:
            if row["case_id"] in ids:
                label_changes[(row["action"], int(row["truth"]), int(row["trunk"]), int(row["final"]))] += int(row["count"])
        groups[name] = dict(paired=paired,
            trunk_dice_percent=100 * float(np.mean([float(r["trunk_dice"]) for r in rows])),
            final_dice_percent=100 * float(np.mean([float(r["final_dice"]) for r in rows])),
            action_voxels={a: sum(int(r[a]) for r in rows) for a in ACTIONS},
            regional_voxels=pooled_regions,
            label_transitions=[dict(action=a, truth=gt, trunk=old, final=new, count=count)
                               for (a, gt, old, new), count in sorted(label_changes.items())])
    files = [directory / name for name in ("summary.json", "patients.csv", "regions.csv", "transitions.csv")]
    result = dict(input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks=dict(patients=55, regional_rows=len(regions), transition_rows=len(transitions),
                              archived_et_match=True, counts_reconcile=True, paired_summary_match=True),
                  groups=groups, losses=[r for r in patients if float(r["delta_pp"]) < 0])
    path = directory / "interpretation-summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=SOURCE / "results/analysis/task-006")
    summarize(parser.parse_args().analysis_dir)
