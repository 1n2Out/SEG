"""用固定 train 划分核验左右镜像参照的标签覆盖率及 FLAIR、T1ce 信号方向。"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from brats.data import load_manifest, write_json
from brats.plotting import write_table


def measure(record, root):
    label = nib.load(root / record["label"])
    flair = nib.load(root / record["images"]["flair"])
    if label.shape != flair.shape or not np.allclose(label.affine, flair.affine, atol=1e-4):
        raise ValueError(f"标签和 FLAIR 几何不一致 {record['id']}")
    axes = nib.aff2axcodes(label.affine)
    lr = next(i for i, axis in enumerate(axes) if axis in ("L", "R"))
    if lr == 2:
        raise ValueError(f"左右轴不在切片平面内 {record['id']}")
    labels = np.asanyarray(label.dataobj)
    wt = labels != 0
    if not wt.any():
        raise ValueError(f"WT 为空 {record['id']}")
    # 只在 WT 坐标取值。镜像关于体积数组中心，属于近似解剖对应。
    coords = np.nonzero(wt)
    opposite = list(coords)
    opposite[lr] = wt.shape[lr] - 1 - coords[lr]
    opposite = tuple(opposite)
    outside = ~wt[opposite]
    image = np.asanyarray(flair.dataobj)
    local, mirror = image[coords], image[opposite]
    valid = outside & (local != 0) & (mirror != 0)
    t1ce = nib.load(root / record["images"]["t1ce"])
    if label.shape != t1ce.shape or not np.allclose(label.affine, t1ce.affine, atol=1e-4):
        raise ValueError(f"T1ce 几何不一致 {record['id']}")
    enhancement = np.asanyarray(t1ce.dataobj)
    et = labels[coords] == 4
    local_ce, mirror_ce = enhancement[coords], enhancement[opposite]
    valid_et = et & outside & (local_ce != 0) & (mirror_ce != 0)
    return {
        "case_id": record["id"], "axes": "".join(axes),
        "wt_voxels": len(local), "mirror_outside_wt_fraction": float(outside.mean()),
        "valid_reference_fraction": float(valid.mean()),
        "flair_brighter_fraction": float((local[valid] > mirror[valid]).mean()) if valid.any() else None,
        "et_voxels": int(et.sum()), "valid_et_voxels": int(valid_et.sum()),
        "et_t1ce_brighter_fraction": float((local_ce[valid_et] > mirror_ce[valid_et]).mean()) if valid_et.any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE / "data/brats2020/split_seed42.json")
    parser.add_argument("--data-root", type=Path, default=SOURCE.parent / "datasets/BraTS2020")
    parser.add_argument("--output-dir", type=Path, default=SOURCE / "results/analysis/contralateral-reference")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    rows = [measure(manifest["cases"][case], args.data_root) for case in manifest["splits"]["train"]]
    result = {"split": "train", "patients": len(rows), "orientations": sorted({r["axes"] for r in rows})}
    for field in ("mirror_outside_wt_fraction", "valid_reference_fraction", "flair_brighter_fraction",
                  "et_t1ce_brighter_fraction"):
        values = [r[field] for r in rows if r[field] is not None]
        result[field] = dict(zip(("p10", "median", "p90"), np.quantile(values, [0.1, 0.5, 0.9]).tolist()))
        result[field]["patients"] = len(values)
    write_table(args.output_dir / "train-per-patient.csv", rows)
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
