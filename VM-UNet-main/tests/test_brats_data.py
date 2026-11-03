import numpy as np
import pytest

from brats.data import (MODALITIES, PREPROCESS_VERSION, SPECIAL_LABELS, check_cache,
                        digest, load_manifest, make_manifest, make_prediction_manifest, normalize_volume,
                        source_signature, validate_manifest, write_json)


def make_case(root, case_id, special=False, labeled=True):
    directory = root / "nested" / case_id
    directory.mkdir(parents=True)
    for modality in MODALITIES:
        (directory / f"{case_id}_{modality}.nii").touch()
    if labeled:
        (directory / (SPECIAL_LABELS[case_id] if special else f"{case_id}_seg.nii")).touch()


def test_patient_split_is_deterministic_disjoint_and_portable(tmp_path):
    for i in range(1, 11):
        make_case(tmp_path, f"BraTS20_Training_{i:03d}")
    make_case(tmp_path, "BraTS20_Validation_001", labeled=False)
    first = make_manifest(tmp_path)
    assert first == make_manifest(tmp_path)
    ids = [case for split in first["splits"].values() for case in split]
    assert len(ids) == len(set(ids)) == 10
    assert first["purpose"] == "training" and "official_val" not in first["splits"]
    prediction = make_prediction_manifest(tmp_path)
    assert prediction["splits"] == {"official_val": ["BraTS20_Validation_001"]}
    assert not set(first["cases"]).intersection(prediction["cases"])
    assert all(not path.startswith(str(tmp_path)) for case in first["cases"].values()
               for path in case["images"].values())
    target = tmp_path / "manifest.json"
    write_json(target, first)
    assert load_manifest(target) == first
    first["splits"]["test"].append(first["splits"]["train"][0])
    first["sha256"] = digest({k: v for k, v in first.items() if k != "sha256"})
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(first)


def test_missing_segmentation_is_not_silently_skipped_and_alias_is_explicit(tmp_path):
    make_case(tmp_path, "BraTS20_Training_001")
    make_case(tmp_path, "BraTS20_Training_002")
    make_case(tmp_path, "BraTS20_Training_355", special=True)
    manifest = make_manifest(tmp_path)
    assert manifest["cases"]["BraTS20_Training_355"]["label"].endswith("W39_1998.09.19_Segm.nii")
    (tmp_path / "nested/BraTS20_Training_002/BraTS20_Training_002_seg.nii").unlink()
    with pytest.raises(ValueError, match="segmentation"):
        make_manifest(tmp_path)


def test_official_cohort_changes_do_not_change_training_manifest(tmp_path):
    for i in range(1, 5):
        make_case(tmp_path, f"BraTS20_Training_{i:03d}")
    training = make_manifest(tmp_path)
    make_case(tmp_path, "BraTS20_Validation_001", labeled=False)
    prediction = make_prediction_manifest(tmp_path)
    assert make_manifest(tmp_path) == training
    # Even an incomplete optional cohort must not block supervised preparation.
    (tmp_path / "nested/BraTS20_Validation_001/BraTS20_Validation_001_t1.nii").unlink()
    assert make_manifest(tmp_path) == training
    with pytest.raises(ValueError, match="t1 NIfTI"):
        make_prediction_manifest(tmp_path)
    # Already-saved training/prediction records remain independently valid.
    validate_manifest(prediction)


def test_prediction_manifest_needs_no_training_directory(tmp_path):
    make_case(tmp_path, "BraTS20_Validation_001", labeled=False)
    prediction = make_prediction_manifest(tmp_path)
    validate_manifest(prediction)
    assert len(prediction["cases"]) == 1
    with pytest.raises(ValueError, match="labeled training"):
        make_manifest(tmp_path)


def test_normalization_keeps_background_zero_and_uses_patient_foreground_only():
    array = np.array([[[0, 0], [1, 2]], [[3, 4], [0, 0]]], dtype=np.float32)
    original = array.copy()
    normalized = normalize_volume(array)
    assert np.array_equal(array, original)
    assert np.all(normalized[array == 0] == 0)
    assert np.isclose(normalized[array != 0].mean(), 0)
    assert np.isclose(normalized[array != 0].std(), 1)


@pytest.mark.parametrize("array", [np.zeros((2, 2, 2)), np.ones((2, 2, 2)),
                                  np.full((2, 2, 2), np.nan)])
def test_bad_modalities_fail(array):
    with pytest.raises(ValueError):
        normalize_volume(array)


def test_cache_rejects_corrupt_shape_and_changed_sources(tmp_path):
    raw = tmp_path / "raw.nii"
    raw.write_bytes(b"original")
    record = {"id": "case", "images": {m: raw.name for m in MODALITIES}, "label": raw.name}
    directory = tmp_path / "cache/case"
    directory.mkdir(parents=True)
    np.save(directory / "image.npy", np.zeros((3, 4, 5, 6), np.float16))
    np.save(directory / "label.npy", np.zeros((3, 5, 6), np.uint8))
    write_json(directory / "metadata.json", {"preprocess_version": PREPROCESS_VERSION,
               "case_id": "case", "modalities": list(MODALITIES),
               "source_signature": source_signature(tmp_path, record), "shape_xyz": [5, 6, 3],
               "spacing_xyz_mm": [1, 1, 1], "spatial_unit": "mm", "affine": np.eye(4).tolist()})
    assert check_cache(tmp_path / "cache", record, tmp_path)["shape_xyz"] == [5, 6, 3]
    np.save(directory / "image.npy", np.zeros((3, 4, 6, 5), np.float16))
    with pytest.raises(ValueError, match="shape/dtype"):
        check_cache(tmp_path / "cache", record, tmp_path)
    raw.write_bytes(b"changed source with different size")
    with pytest.raises(ValueError, match="Source files changed"):
        check_cache(tmp_path / "cache", record, tmp_path)
    np.save(directory / "image.npy", np.zeros((3, 4, 5, 6), np.float16))
    raw.unlink()
    assert check_cache(tmp_path / "cache", record)["shape_xyz"] == [5, 6, 3]


def test_cache_geometry_is_checked_without_raw_files(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    np.save(directory / "image.npy", np.zeros((3, 4, 5, 6), np.float16))
    record = {"id": "case", "images": {m: "missing.nii" for m in MODALITIES}}
    metadata = {"case_id": "case", "modalities": list(MODALITIES),
                "preprocess_version": PREPROCESS_VERSION, "shape_xyz": [5, 6, 3],
                "spacing_xyz_mm": [1, 1, 2], "spatial_unit": "mm", "affine": np.eye(4).tolist()}
    write_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError, match="spacing disagree"):
        check_cache(tmp_path, record)
