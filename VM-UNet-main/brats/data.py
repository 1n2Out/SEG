"""Portable patient manifests and explicit, disk-backed preprocessing.

Inventory/splitting needs only Python. NIfTI preprocessing additionally needs
numpy and nibabel; PyTorch is deliberately not imported here.
"""
import hashlib
import json
import random
from pathlib import Path

MODALITIES = ("t1", "t1ce", "t2", "flair")
SPECIAL_LABELS = {"BraTS20_Training_355": "W39_1998.09.19_Segm.nii"}
PREPROCESS_VERSION = "brats2020-nonzero-zscore-fp16-xyz-to-zcxy-v2"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def discover_cases(root, labeled=True):
    root = Path(root).resolve()
    result = []
    prefix = "BraTS20_Training_" if labeled else "BraTS20_Validation_"
    directories = sorted(p for p in root.rglob(prefix + "*") if p.is_dir())
    for directory in directories:
        record = {"id": directory.name, "images": {}}
        for modality in MODALITIES:
            candidates = [p for p in directory.glob(directory.name + "_" + modality + ".nii*")
                          if p.name.endswith((".nii", ".nii.gz"))]
            if len(candidates) != 1:
                raise ValueError(f"{directory}: expected one {modality} NIfTI, found {candidates}")
            record["images"][modality] = candidates[0].relative_to(root).as_posix()
        if labeled:
            labels = [p for p in directory.glob(directory.name + "_seg.nii*")
                      if p.name.endswith((".nii", ".nii.gz"))]
            if not labels and directory.name in SPECIAL_LABELS:
                alternative = directory / SPECIAL_LABELS[directory.name]
                labels = [alternative] if alternative.is_file() else []
            if len(labels) != 1:
                raise ValueError(f"{directory}: missing or ambiguous segmentation: {labels}")
            record["label"] = labels[0].relative_to(root).as_posix()
        result.append(record)
    if len(result) < (3 if labeled else 1):
        raise ValueError(f"Not enough {'labeled training' if labeled else 'official validation'} patients under {root}")
    ids = [r["id"] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs; provide one dataset root, without duplicate copies")
    return result


def make_manifest(root, seed=42, val_fraction=0.15, test_fraction=0.15):
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and
            val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    records = discover_cases(root)
    ids = [r["id"] for r in records]
    random.Random(seed).shuffle(ids)
    nv, nt = max(1, round(len(ids) * val_fraction)), max(1, round(len(ids) * test_fraction))
    if nv + nt >= len(ids):
        raise ValueError("Not enough patients for the requested split")
    splits = {"val": sorted(ids[:nv]), "test": sorted(ids[nv:nv + nt]),
              "train": sorted(ids[nv + nt:])}
    manifest = {"version": 2, "purpose": "training", "seed": seed, "val_fraction": val_fraction,
                "test_fraction": test_fraction, "modalities": list(MODALITIES),
                "raw_labels": [0, 1, 2, 4], "splits": splits,
                "cases": {r["id"]: r for r in records}}
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def make_prediction_manifest(root):
    records = discover_cases(root, labeled=False)
    manifest = {"version": 2, "purpose": "prediction", "modalities": list(MODALITIES),
                "raw_labels": [0, 1, 2, 4],
                "splits": {"official_val": [r["id"] for r in records]},
                "cases": {r["id"]: r for r in records}}
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("sha256") != digest(unsigned):
        raise ValueError("Manifest checksum mismatch")
    if manifest["modalities"] != list(MODALITIES):
        raise ValueError("Unexpected modality order")
    if manifest.get("version") != 2 or manifest.get("purpose") not in ("training", "prediction"):
        raise ValueError("Expected a v2 training or prediction manifest")
    splits = ("train", "val", "test") if manifest["purpose"] == "training" else ("official_val",)
    if set(manifest["splits"]) != set(splits):
        raise ValueError("Training and official prediction splits must use separate manifests")
    seen = set()
    for split in splits:
        ids = manifest["splits"][split]
        if not ids:
            raise ValueError(f"Empty {split} split")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Patient overlap or duplicate IDs in split manifest")
        seen.update(ids)
        for case_id in ids:
            case = manifest["cases"][case_id]
            if case["id"] != case_id or (split != "official_val" and "label" not in case):
                raise ValueError(f"Invalid labeled case {case_id}")
    if seen != set(manifest["cases"]):
        raise ValueError("Manifest has unassigned cases")


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def source_signature(root, record):
    paths = list(record["images"].values()) + ([record["label"]] if "label" in record else [])
    return digest({name: [Path(root, name).stat().st_size, Path(root, name).stat().st_mtime_ns]
                   for name in paths})


def check_cache(cache_dir, record, root=None):
    import numpy as np
    directory = Path(cache_dir) / record["id"]
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["preprocess_version"] != PREPROCESS_VERSION:
        raise ValueError(f"Stale preprocessing cache: {directory}")
    if metadata["case_id"] != record["id"] or metadata["modalities"] != list(MODALITIES):
        raise ValueError(f"Wrong case or modality order in cache: {directory}")
    if root is not None and metadata["source_signature"] != source_signature(root, record):
        raise ValueError(f"Source files changed; run prepare_brats.py --cache --rebuild-cache: {directory}")
    required = [directory / "image.npy"]
    if "label" in record:
        required.append(directory / "label.npy")
    if not all(p.is_file() for p in required):
        raise FileNotFoundError(f"Incomplete cache: {directory}; run prepare_brats.py --cache")
    x, y, z = metadata["shape_xyz"]
    if any(not isinstance(s, int) or s <= 0 for s in (x, y, z)):
        raise ValueError(f"Invalid cache volume dimensions: {directory}")
    affine = np.asarray(metadata["affine"], dtype=float)
    spacing = np.asarray(metadata["spacing_xyz_mm"], dtype=float)
    unit_scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(metadata["spatial_unit"])
    if (affine.shape != (4, 4) or not np.isfinite(affine).all() or
            spacing.shape != (3,) or not np.isfinite(spacing).all() or
            (spacing <= 0).any() or unit_scale is None):
        raise ValueError(f"Invalid cached spatial geometry: {directory}")
    if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0) * unit_scale, spacing, atol=1e-4):
        raise ValueError(f"Cached affine and millimeter spacing disagree: {directory}")
    image = np.load(directory / "image.npy", mmap_mode="r", allow_pickle=False)
    if image.shape != (z, 4, x, y) or image.dtype != np.float16:
        raise ValueError(f"Invalid image cache shape/dtype: {directory}")
    if "label" in record:
        label = np.load(directory / "label.npy", mmap_mode="r", allow_pickle=False)
        if label.shape != (z, x, y) or label.dtype != np.uint8:
            raise ValueError(f"Invalid label cache shape/dtype: {directory}")
    return metadata


def normalize_volume(array):
    import numpy as np
    array = np.asarray(array, dtype=np.float32).copy()
    if not np.isfinite(array).all():
        raise ValueError("MRI contains nonfinite voxels")
    foreground = array != 0
    if not foreground.any():
        raise ValueError("MRI modality is entirely empty")
    values = array[foreground]
    std = float(values.std())
    if std < 1e-8:
        raise ValueError("MRI modality has constant nonzero intensity")
    array[foreground] = (values - float(values.mean())) / std
    array[~foreground] = 0
    return array


def preprocess_case(root, cache_dir, record, rebuild=False):
    """Keep voxel geometry; cache image Z,C,X,Y and raw label X,Y,Z -> Z,X,Y.

    Each modality is z-scored over its nonzero voxels within that patient.
    No fitted statistics, annotation-derived cropping, resizing, or skull stripping.
    """
    import numpy as np
    import nibabel as nib
    directory = Path(cache_dir) / record["id"]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "metadata.json"
    if marker.exists() and not rebuild:
        check_cache(cache_dir, record, root)
        return "cached"
    if marker.exists():
        marker.unlink()  # invalidate completion marker before replacing arrays
    reference = None
    images = []
    for modality in MODALITIES:
        image = nib.load(str(Path(root) / record["images"][modality]))
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError(f"Invalid 3D geometry: {record['id']}/{modality}")
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Unaligned MRI modalities: {record['id']}/{modality}")
        images.append(normalize_volume(image.get_fdata(dtype=np.float32)).astype(np.float16))
    units = reference.header.get_xyzt_units()[0]
    unit_scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
    if unit_scale is None:
        raise ValueError(f"Unknown spatial units {units}: cannot report HD95 in mm")
    spacing = np.sqrt((reference.affine[:3, :3] ** 2).sum(axis=0)) * unit_scale
    directions = reference.affine[:3, :3] / (spacing / unit_scale)
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("Sheared voxel grid requires resampling before Euclidean HD95")
    label_info = None
    if "label" in record:
        label_image = nib.load(str(Path(root) / record["label"]))
        if label_image.shape != reference.shape or not np.allclose(label_image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Image/label geometry mismatch: {record['id']}")
        label = label_image.get_fdata(dtype=np.float32)
        values, counts = np.unique(label, return_counts=True)
        if not np.isin(values, [0, 1, 2, 4]).all():
            raise ValueError(f"Invalid BraTS labels {values}: {record['id']}")
        label_info = {str(int(v)): int(n) for v, n in zip(values, counts)}
        _save_array(directory / "label.npy", label.transpose(2, 0, 1).astype(np.uint8))
    image_array = np.stack(images, axis=0).transpose(3, 0, 1, 2)
    if not np.isfinite(image_array).all():
        raise ValueError("Normalization overflow in float16 cache")
    _save_array(directory / "image.npy", image_array)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    metadata = {"case_id": record["id"], "modalities": list(MODALITIES),
                "preprocess_version": PREPROCESS_VERSION,
                "source_signature": source_signature(root, record),
                "shape_xyz": list(reference.shape), "spacing_xyz_mm": spacing.tolist(),
                "affine": reference.affine.tolist(), "spatial_unit": units,
                "qform": qform.tolist() if qform is not None else None, "qform_code": int(qcode),
                "sform": sform.tolist() if sform is not None else None, "sform_code": int(scode),
                "label_voxel_counts": label_info}
    # Metadata is the completion marker and is written only after both arrays.
    write_json(directory / "metadata.json", metadata)
    return "prepared"


def save_prediction(prediction, metadata, path):
    """Export original-label XYZ segmentation using cached geometry only."""
    import numpy as np
    import nibabel as nib
    if tuple(prediction.shape) != tuple(metadata["shape_xyz"]) or not np.isin(prediction, [0, 1, 2, 4]).all():
        raise ValueError("Prediction must match cached XYZ shape and original BraTS labels")
    output = nib.Nifti1Image(prediction.astype(np.uint8), np.asarray(metadata["affine"]))
    output.header.set_xyzt_units(metadata["spatial_unit"])
    output.set_qform(metadata["qform"], code=metadata["qform_code"])
    output.set_sform(metadata["sform"], code=metadata["sform_code"])
    output.header.set_slope_inter(1.0, 0.0)
    output.header["cal_min"], output.header["cal_max"] = 0, 4
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(path))


def _save_array(path, array):
    import numpy as np
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)
