"""Patient-level, overlapping WT/TC/ET metrics on raw-label XYZ volumes.

This is an explicit local protocol, not an assertion of official server parity.
Both masks empty: Dice/precision/recall = 1, HD95 = 0.
Exactly one empty: Dice/precision/recall = 0, HD95 = physical grid diagonal.
HD95 pools the two directed surface-distance lists (6-connected surfaces).
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

REGIONS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}
METRICS = ("dice", "hd95_mm", "precision", "recall", "specificity")
PROTOCOL = {
    "name": "local-brats2020-volume-v1", "regions": REGIONS,
    "aggregation": "per-patient 3D, then unweighted mean across patients",
    "dice_scale": "0..1", "hd95": "pooled bidirectional 6-connected surface distances in mm",
    "both_empty": "dice=precision=recall=1; hd95_mm=0",
    "one_empty": "dice=precision=recall=0; hd95_mm=norm(shape_xyz * spacing_xyz_mm)",
    "std": "population standard deviation across patients (ddof=0)",
}


def binary_metrics(prediction, target, spacing):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Expected equal 3D XYZ masks")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Spacing must contain three finite positive millimeter values")
    npred, ntrue = int(prediction.sum()), int(target.sum())
    tp = int(np.count_nonzero(prediction & target))
    fp = npred - tp
    tn = prediction.size - ntrue - fp
    specificity = tn / (prediction.size - ntrue) if prediction.size != ntrue else 1.0
    if npred == 0 or ntrue == 0:
        both_empty = npred == ntrue == 0
        score = float(both_empty)
        return {"dice": score, "hd95_mm": 0.0 if both_empty else float(np.linalg.norm(np.array(target.shape) * spacing)),
                "precision": score, "recall": score, "specificity": specificity}
    # Distances need only the union bounding box; preserve one background voxel
    # around it so crop boundaries do not alter surfaces.
    coordinates = np.where(prediction | target)
    roi = tuple(slice(max(0, int(c.min()) - 1), min(s, int(c.max()) + 2))
                for c, s in zip(coordinates, target.shape))
    p, t = prediction[roi], target[roi]
    structure = generate_binary_structure(3, 1)
    ps = p ^ binary_erosion(p, structure=structure, border_value=0)
    ts = t ^ binary_erosion(t, structure=structure, border_value=0)
    pt = distance_transform_edt(~ts, sampling=spacing)[ps]
    tpred = distance_transform_edt(~ps, sampling=spacing)[ts]
    return {"dice": 2 * tp / (npred + ntrue),
            "hd95_mm": float(np.percentile(np.concatenate([pt, tpred]), 95)),
            "precision": tp / npred, "recall": tp / ntrue, "specificity": specificity}


def evaluate_regions(prediction, target, spacing):
    for volume in (prediction, target):
        if not np.isin(volume, [0, 1, 2, 4]).all():
            raise ValueError("Metrics expect original BraTS labels 0,1,2,4")
    return {region: binary_metrics(np.isin(prediction, labels), np.isin(target, labels), spacing)
            for region, labels in REGIONS.items()}


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    summary = {"n_patients": len(rows), "protocol": PROTOCOL, "regions": {}}
    for region in REGIONS:
        summary["regions"][region] = {}
        for metric in METRICS:
            values = np.array([row[region][metric] for row in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite metric encountered")
            summary["regions"][region][metric] = {"mean": float(values.mean()), "std_patients": float(values.std())}
    summary["mean_dice"] = float(np.mean([summary["regions"][r]["dice"]["mean"] for r in REGIONS]))
    summary["mean_hd95_mm"] = float(np.mean([summary["regions"][r]["hd95_mm"]["mean"] for r in REGIONS]))
    return summary
