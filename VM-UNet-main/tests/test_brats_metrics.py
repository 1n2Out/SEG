import numpy as np
import pytest
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial.distance import cdist

from brats.metrics import REGIONS, binary_metrics, evaluate_regions, summarize


def test_perfect_raw_labels_and_nested_regions():
    target = np.zeros((8, 8, 8), np.uint8)
    target[1:3, 1:3, 1:3] = 1
    target[3:5, 3:5, 3:5] = 2
    target[5:7, 5:7, 5:7] = 4
    scores = evaluate_regions(target, target.copy(), (1, 1, 1))
    for region in REGIONS:
        assert scores[region]["dice"] == 1
        assert scores[region]["hd95_mm"] == 0
    edema_only = np.where(target == 2, 2, 0)
    partial = evaluate_regions(edema_only, target, (1, 1, 1))
    assert partial["WT"]["dice"] == pytest.approx(0.5)
    assert partial["TC"]["dice"] == 0
    assert partial["ET"]["dice"] == 0


def test_empty_regions_are_scored_and_not_dropped():
    empty = np.zeros((4, 5, 6), bool)
    full = empty.copy()
    full[1, 2, 3] = True
    both = binary_metrics(empty, empty, (1, 2, 3))
    assert both["dice"] == 1 and both["hd95_mm"] == 0
    for p, t in ((empty, full), (full, empty)):
        score = binary_metrics(p, t, (1, 2, 3))
        assert score["dice"] == 0
        assert score["hd95_mm"] == pytest.approx(np.linalg.norm([4, 10, 18]))


def test_hd95_respects_xyz_spacing():
    p, t = np.zeros((5, 5, 5), bool), np.zeros((5, 5, 5), bool)
    p[2, 1, 2], t[2, 2, 2] = True, True
    assert binary_metrics(p, t, (1, 2, 3))["hd95_mm"] == pytest.approx(2)


def test_cropped_hd95_matches_independent_pairwise_surface_distances():
    p, t = np.zeros((15, 12, 10), bool), np.zeros((15, 12, 10), bool)
    p[0:4, 3:7, 2:5] = True  # includes a boundary-touching object
    t[1:6, 2:6, 3:7] = True
    structure = generate_binary_structure(3, 1)
    ps = np.argwhere(p ^ binary_erosion(p, structure=structure, border_value=0))
    ts = np.argwhere(t ^ binary_erosion(t, structure=structure, border_value=0))
    spacing = np.array([1.5, 2, 3])
    distances = cdist(ps * spacing, ts * spacing)
    expected = np.percentile(np.r_[distances.min(axis=0), distances.min(axis=1)], 95)
    assert binary_metrics(p, t, spacing)["hd95_mm"] == pytest.approx(expected)


def test_summary_is_patient_macro_average_not_voxel_pooling():
    small = np.zeros((5, 5, 5), np.uint8)
    small[1, 1, 1] = 4
    large = np.ones((5, 5, 5), np.uint8) * 4
    rows = [evaluate_regions(small, small, (1, 1, 1)),
            evaluate_regions(np.zeros_like(large), large, (1, 1, 1))]
    summary = summarize(rows)
    assert summary["n_patients"] == 2
    assert summary["mean_dice"] == pytest.approx(0.5)
    assert summary["regions"]["ET"]["dice"]["std_patients"] == pytest.approx(0.5)


def test_invalid_labels_and_spacing_fail():
    with pytest.raises(ValueError, match="original BraTS labels"):
        evaluate_regions(np.ones((2, 2, 2)) * 3, np.zeros((2, 2, 2)), (1, 1, 1))
    with pytest.raises(ValueError, match="Spacing"):
        binary_metrics(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), (1, 0, 1))
