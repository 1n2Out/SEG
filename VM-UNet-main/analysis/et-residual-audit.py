"""TASK-006：读取 EXP-006 val best、固定 val 缓存及归档逐患者 CSV。

一次前向收集残差前后预测，逐例核验，产物默认写入 results/analysis/task-006/。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from brats.data import check_cache, load_manifest, write_json
from brats.metrics import REGIONS
from brats.plotting import write_table
from brats.provenance import git_state

ARCHIVE = SOURCE.parent / "research/experiments/EXP-006"
ZONES = ("interior", "inner-boundary", "outer-boundary", "outside")
ACTIONS = ("recover", "remove-fp", "lose", "add-fp")


def predict_pair(model, images, device, config):
    """Capture the final projection of the trained trunk during normal inference."""
    from brats.dataset import padding
    from brats.engine import predict_volume
    import torch

    size = config["input_size"]
    n, _, x, y = images.shape
    (left, _), (top, _) = padding((x, y), size)
    batches = []

    def capture(module, inputs, logits):
        if logits.shape[1:] != (4, size, size) or not torch.isfinite(logits).all():
            raise ValueError("Invalid trunk logits")
        indices = logits[:, :, left:left+x, top:top+y].argmax(dim=1).byte().cpu().numpy()
        indices[indices == 3] = 4
        batches.append(indices)

    hook = model.vmunet.final_conv.register_forward_hook(capture)
    try:
        final = predict_volume(model, images, device, size, config["inference_batch_size"], config["amp"])
    finally:
        hook.remove()
    before = np.concatenate(batches)
    if before.shape != (n, x, y):
        raise ValueError("Expected exactly one trunk prediction per slice")
    return before.transpose(1, 2, 0), final


def dice(a, b):
    denominator = int(a.sum()) + int(b.sum())
    return 2 * np.count_nonzero(a & b) / denominator if denominator else 1.0


def verify_final(case_id, target, final, reference):
    for region, labels in REGIONS.items():
        truth, prediction = np.isin(target, labels), np.isin(final, labels)
        for suffix, actual in (("target_voxels", int(truth.sum())),
                               ("prediction_voxels", int(prediction.sum())),
                               ("dice", dice(truth, prediction))):
            key = f"{region}_{suffix}"
            expected = float(reference[key])
            if not np.isclose(actual, expected, atol=1e-12, rtol=0):
                raise ValueError(f"{case_id} {key}: reproduced {actual}, archive {expected}")


def analyze_case(case_id, target, before, final):
    truth, prior, post = target == 4, before == 4, final == 4
    prior_dice, final_dice = dice(truth, prior), dice(truth, post)
    patient = dict(case_id=case_id, target_et=int(truth.sum()), trunk_et=int(prior.sum()),
                   final_et=int(post.sum()), trunk_dice=prior_dice, final_dice=final_dice,
                   delta_pp=100 * (final_dice - prior_dice))
    patient["trunk_et_present"] = patient["trunk_et"] > 0
    patient["final_et_present"] = patient["final_et"] > 0
    changed = prior ^ post
    slice_index = int(changed.sum(axis=(0, 1)).argmax())
    patient["slice_z"] = slice_index
    # Zones partition each axial plane. Empty ET cases get a separate category.
    zone = np.full(target.shape, 3, dtype=np.uint8)
    names = ZONES
    if truth.any():
        kernel = np.ones((3, 3, 1), dtype=bool)
        zone[binary_dilation(truth, structure=kernel)] = 2
        zone[truth] = 1
        zone[binary_erosion(truth, structure=kernel)] = 0
    else:
        zone.fill(0)
        names = ("empty-et",)

    rates, transitions = [], []
    opportunities = (truth & ~prior, ~truth & prior, truth & prior, ~truth & ~prior)
    for action, eligible in zip(ACTIONS, opportunities):
        flipped = eligible & changed
        patient[action] = int(flipped.sum())
        totals = np.bincount(zone[eligible], minlength=len(names))
        counts = np.bincount(zone[flipped], minlength=len(names))
        for index, name in enumerate(names):
            rates.append(dict(case_id=case_id, zone=name, action=action, count=int(counts[index]),
                              opportunities=int(totals[index]),
                              rate=float(counts[index] / totals[index]) if totals[index] else None))
        # Only voxels whose ET membership changed contribute a label transition.
        codes = np.stack([zone[flipped], target[flipped], before[flipped], final[flipped]], axis=1)
        values, frequencies = np.unique(codes, axis=0, return_counts=True)
        for (index, gt, old, new), frequency in zip(values, frequencies):
            transitions.append(dict(case_id=case_id, zone=names[index], action=action,
                                    truth=int(gt), trunk=int(old), final=int(new), count=int(frequency)))
    if sum(patient[key] for key in ACTIONS) != np.count_nonzero(changed):
        raise ValueError(f"{case_id}: ET flip counts do not reconcile")
    return patient, rates, transitions


def select_cases(positive):
    ordered = sorted(positive, key=lambda row: row["case_id"])
    median = np.median([row["delta_pp"] for row in ordered])
    gains = [row for row in ordered if row["delta_pp"] > 0]
    losses = [row for row in ordered if row["delta_pp"] < 0]
    return {"gain": max(gains, key=lambda r: r["delta_pp"])["case_id"] if gains else None,
            "typical": min(ordered, key=lambda r: abs(r["delta_pp"] - median))["case_id"],
            "loss": min(losses, key=lambda r: r["delta_pp"])["case_id"] if losses else None}


def write_results(output, patients, rates, transitions, slices, provenance):
    positive = [row for row in patients if row["target_et"]]
    empty = [row for row in patients if not row["target_et"]]
    groups = {}
    for name, rows in (("all", patients), ("nonempty", positive), ("empty", empty)):
        delta = np.array([row["delta_pp"] for row in rows])
        groups[name] = dict(n=len(rows), mean_pp=float(delta.mean()), median_pp=float(np.median(delta)),
                            gains=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    regional = []
    for zone in (*ZONES, "empty-et"):
        for action in ACTIONS:
            values = [r["rate"] for r in rates if r["zone"] == zone and r["action"] == action and r["rate"] is not None]
            regional.append(dict(zone=zone, action=action, valid_patients=len(values),
                                 mean_rate=float(np.mean(values)) if values else None))
    selected = select_cases(positive)
    write_table(output / "patients.csv", patients)
    write_table(output / "regions.csv", rates)
    write_table(output / "transitions.csv", transitions,
                fields=["case_id", "zone", "action", "truth", "trunk", "final", "count"])
    for case_id in sorted(set(selected.values()) - {None}):
        np.savez_compressed(output / (case_id.lower().replace("_", "-") + ".npz"), **slices[case_id])
    write_json(output / "summary.json", dict(provenance=provenance, paired=groups,
               selected=selected, regional_rates=regional, empty_et=empty))


def run(args, output):
    import torch
    from train_brats import build_model, load_checkpoint

    manifest = load_manifest(SOURCE / "data/brats2020/split_seed42.json")
    ids = manifest["splits"]["val"]
    csv_path = ARCHIVE / "val_per_patient.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reference = list(csv.DictReader(stream))
    by_id = {row["case_id"]: row for row in reference}
    if len(reference) != 55 or len(ids) != 55 or set(ids) != by_id.keys():
        raise ValueError("Archive and validation manifest must match for all 55 unique patients")
    record = json.loads((ARCHIVE / "summary.json").read_text(encoding="utf-8"))
    checkpoint_path = SOURCE / "results" / record["run_dir"] / "best.pth"
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if (checkpoint["manifest_sha256"] != manifest["sha256"] or
        checkpoint["epoch"] != record["selected_epoch"] or
        config != json.loads((ARCHIVE / "config.json").read_text(encoding="utf-8"))):
        raise ValueError("Checkpoint split, epoch or config differs from EXP-006")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config["seed"])
    device = torch.device("cuda:0")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device).eval()
    patients, rates, transitions, slices = [], [], [], {}
    for i, case_id in enumerate(ids, 1):
        check_cache(args.cache_dir, manifest["cases"][case_id])
        directory = args.cache_dir / case_id
        images = np.load(directory / "image.npy", mmap_mode="r")
        target = np.load(directory / "label.npy", mmap_mode="r").transpose(1, 2, 0)
        before, final = predict_pair(model, images, device, config)
        verify_final(case_id, target, final, by_id[case_id])
        patient, areas, flips = analyze_case(case_id, target, before, final)
        patients.append(patient)
        rates.extend(areas)
        transitions.extend(flips)
        if patient["target_et"]:
            z = patient["slice_z"]
            slices[case_id] = dict(image=images[z, 1].copy(), truth=target[:, :, z].copy(),
                                   trunk=before[:, :, z].copy(), final=final[:, :, z].copy(), z=z)
        print(f"[val {i}/{len(ids)}] {case_id} verified", flush=True)
    if len(slices) != 50:
        raise ValueError("Expected 50 nonempty ET patients")
    write_results(output, patients, rates, transitions, slices, {
        "git": git_state(SOURCE), "split": "val", "manifest_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint_path), "epoch": record["selected_epoch"],
        "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
        "reference_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, choices=("2", "3", "4", "5"), help="Single physical RTX 6000 GPU")
    parser.add_argument("--cache-dir", type=Path, default=SOURCE / "data/brats2020/cache")
    parser.add_argument("--output", type=Path, default=SOURCE / "results/analysis/task-006")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.output.exists():
        parser.error("Output already exists; specify a fresh --output directory for a new run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        directory = Path(staging) / "verified"
        directory.mkdir()
        run(args, directory)
        directory.rename(args.output)
    print(f"Verified results: {args.output}")


if __name__ == "__main__":
    main()
