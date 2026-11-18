"""TASK-006 CPU checks: paired inference, regional transitions and offline figures."""
import csv
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from brats.engine import predict_volume
from plot_brats import main as plot

SOURCE = Path(__file__).resolve().parents[1]
audit = runpy.run_path(str(SOURCE / "analysis/et-residual-audit.py"))


def test_paired_capture_preserves_native_prediction_and_one_forward_per_batch():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vmunet = nn.Module()
            self.vmunet.final_conv = nn.Identity()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return self.vmunet.final_conv(batch).roll(1, dims=1)

    images = np.random.default_rng(42).normal(size=(5, 4, 17, 21)).astype(np.float16)
    config = dict(input_size=32, inference_batch_size=2, amp=False)
    model, device = ToyModel(), torch.device("cpu")
    before, final = audit["predict_pair"](model, images, device, config)
    assert model.calls == 3
    assert not model.vmunet.final_conv._forward_hooks
    expected = np.array([0, 1, 2, 4])[images.argmax(axis=1)].transpose(1, 2, 0)
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(final, predict_volume(model, images, device, 32, 2))


def test_zones_actions_and_reference():
    target = np.zeros((7, 7, 3), dtype=np.uint8)
    target[2:5, 2:5, 1] = 4
    target[1, 2, 1] = 1
    before, final = target.copy(), target.copy()
    before[2, 2, 1] = 1   # inner band FN -> TP
    final[3, 3, 1] = 1    # interior TP -> FN
    before[1, 2, 1] = 4   # outer band FP -> TN
    final[3, 3, 0] = 4    # outside (no axial mixing) TN -> FP
    case, regions, transitions = audit["analyze_case"]("p", target, before, final)
    assert all(case[action] == 1 for action in audit["ACTIONS"])
    assert case["slice_z"] == 1
    counts = {(r["zone"], r["action"]): r for r in regions}
    assert counts["inner-boundary", "recover"]["rate"] == 1
    assert counts["interior", "lose"]["rate"] == 1
    assert counts["outer-boundary", "remove-fp"]["rate"] == 1
    assert counts["outside", "add-fp"]["count"] == 1
    assert counts["interior", "add-fp"]["rate"] is None
    assert len(transitions) == 4
    assert any(t["truth"] == 1 and t["trunk"] == 4 and t["final"] == 1 for t in transitions)
    # WT and TC retain the voxel converted from ET to core, plus one outside FP.
    reference = {"WT_target_voxels": 10, "WT_prediction_voxels": 11, "WT_dice": 20/21,
                 "TC_target_voxels": 10, "TC_prediction_voxels": 11, "TC_dice": 20/21,
                 "ET_target_voxels": 9, "ET_prediction_voxels": 9, "ET_dice": 8/9}
    audit["verify_final"]("p", target, final, reference)
    with pytest.raises(ValueError, match="p ET_dice"):
        audit["verify_final"]("p", target, final, {**reference, "ET_dice": 1})


def test_export_empty_et_and_figures(tmp_path):
    empty = np.zeros((5, 5, 2), dtype=np.uint8)
    lesion = empty.copy()
    lesion[2, 2, :] = 4
    patients, rates, transitions, slices = [], [], [], {}
    for case_id, truth, before, final in (("a", lesion, empty, lesion),
                                          ("b", lesion, lesion, empty),
                                          ("c", empty, empty, lesion)):
        case, zones, flips = audit["analyze_case"](case_id, truth, before, final)
        patients.append(case)
        rates.extend(zones)
        transitions.extend(flips)
        assert case["slice_z"] == 0
        slices[case_id] = dict(image=empty[:, :, 0], truth=truth[:, :, 0],
                               trunk=before[:, :, 0], final=final[:, :, 0], z=0)
    audit["write_results"](tmp_path, patients, rates, transitions, slices, {})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["selected"] == {"gain": "a", "typical": "a", "loss": "b"}
    assert summary["empty_et"][0]["delta_pp"] == -100
    assert summary["empty_et"][0]["trunk_et_present"] is False
    assert summary["empty_et"][0]["final_et_present"] is True
    assert summary["paired"]["nonempty"]["median_pp"] == 0
    with (tmp_path / "regions.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert any(r["rate"] == "" for r in csv.DictReader(stream))
    plot(["et-residual-audit", "--analysis-dir", str(tmp_path), "--formats", "png"])
    assert {p.stem for p in (tmp_path / "figures").glob("*.png")} == {
        "regional-rates", "case-gain", "case-typical", "case-loss"}
    assert audit["select_cases"]([{"case_id": "p", "delta_pp": 0}]) == {
        "gain": None, "typical": "p", "loss": None}
