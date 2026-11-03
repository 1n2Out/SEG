"""Synthetic NIfTI geometry and raw-label export tests."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel", reason="NIfTI tests need nibabel in the training environment")
from brats.data import MODALITIES, check_cache, preprocess_case, save_prediction


def create_niftis(root):
    images = {}
    affine = np.diag([-1., -1., 2., 1.])
    for i, modality in enumerate(MODALITIES):
        data = np.arange(120, dtype=np.float32).reshape(4, 5, 6) + i
        image = nib.Nifti1Image(data, affine)
        image.header.set_xyzt_units("mm")
        path = root / f"case_{modality}.nii.gz"
        nib.save(image, path)
        images[modality] = path.name
    label = np.zeros((4, 5, 6), np.uint8)
    label[1:3, 2:4, 1:3] = 4
    nib.save(nib.Nifti1Image(label, affine), root / "case_seg.nii.gz")
    return {"id": "case", "images": images, "label": "case_seg.nii.gz"}, label


def test_preprocessing_preserves_geometry_and_label_values(tmp_path):
    record, label = create_niftis(tmp_path)
    cache = tmp_path / "cache"
    assert preprocess_case(tmp_path, cache, record) == "prepared"
    metadata = check_cache(cache, record, tmp_path)
    assert metadata["spacing_xyz_mm"] == [1., 1., 2.]
    assert np.load(cache / "case/image.npy").shape == (6, 4, 4, 5)
    assert np.array_equal(np.load(cache / "case/label.npy"), label.transpose(2, 0, 1))
    assert preprocess_case(tmp_path, cache, record) == "cached"


def test_misaligned_label_is_rejected(tmp_path):
    record, label = create_niftis(tmp_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), tmp_path / record["label"])
    with pytest.raises(ValueError, match="geometry mismatch"):
        preprocess_case(tmp_path, tmp_path / "cache", record)


@pytest.mark.parametrize("spatial_unit,scale", [("mm", 1.0), ("meter", 1000.0)])
@pytest.mark.parametrize("form", ["sform", "qform"])
def test_export_works_after_original_niftis_are_removed(tmp_path, spatial_unit, scale, form):
    record, label = create_niftis(tmp_path)
    expected_affine = np.diag([-1., -1., 2., 1.])
    expected_affine[:3] /= scale
    expected_affine[3, 3] = 1
    for name in [*record["images"].values(), record["label"]]:
        path = tmp_path / name
        image = nib.load(path)
        data = image.get_fdata().copy()
        image = nib.Nifti1Image(data, expected_affine)
        image.header.set_xyzt_units(spatial_unit)
        image.set_qform(expected_affine if form == "qform" else None, code=1 if form == "qform" else 0)
        image.set_sform(expected_affine if form == "sform" else None, code=2 if form == "sform" else 0)
        nib.save(image, path)
    cache = tmp_path / "cache"
    preprocess_case(tmp_path, cache, record)
    metadata = check_cache(cache, record)
    assert np.allclose(metadata["spacing_xyz_mm"], [1, 1, 2])
    for name in [*record["images"].values(), record["label"]]:
        (tmp_path / name).unlink()
    check_cache(cache, record)
    output = tmp_path / "export/prediction.nii.gz"
    save_prediction(label, metadata, output)
    saved = nib.load(output)
    assert np.array_equal(saved.get_fdata(), label)
    assert np.allclose(saved.affine, expected_affine, atol=1e-6)
    assert saved.header.get_xyzt_units()[0] == spatial_unit
    assert saved.get_data_dtype() == np.dtype(np.uint8)
    assert int(saved.header["qform_code"]) == (1 if form == "qform" else 0)
    assert int(saved.header["sform_code"]) == (2 if form == "sform" else 0)
