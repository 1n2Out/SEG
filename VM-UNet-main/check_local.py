"""CPU development checks; no VM-UNet scan, backward, optimizer or training."""
import argparse
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_modules():
    import torch
    from brats.losses import BraTSCEDiceLoss
    from models.vmunet.vmamba import (Final_PatchExpand2D, PatchEmbed2D,
                                     PatchExpand2D, PatchMerging2D)
    from models.vmunet.vmunet import VMUNet  # Import only; no full model forward.

    torch.manual_seed(0)
    with torch.no_grad():
        embedded = PatchEmbed2D(in_chans=4, embed_dim=8)(torch.randn(1, 4, 32, 32))
        merged = PatchMerging2D(dim=8)(embedded)
        expanded = PatchExpand2D(dim=8)(merged)
        final = Final_PatchExpand2D(dim=8)(expanded)
        expected = [(1, 8, 8, 8), (1, 4, 4, 16), (1, 8, 8, 8), (1, 32, 32, 2)]
        for tensor, shape in zip((embedded, merged, expanded, final), expected):
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise RuntimeError(f"Invalid ordinary-module output: {tensor.shape}")
        logits = torch.nn.Conv2d(2, 4, 1)(final.permute(0, 3, 1, 2))
        loss = BraTSCEDiceLoss()(logits, torch.zeros(1, 32, 32, dtype=torch.long))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite forward-only loss")
    return {"patch_shapes": [list(s) for s in expected], "forward_only_loss": loss.item(),
            "model_import": VMUNet.__name__, "device": "cpu"}


def check_sample(args):
    import nibabel as nib
    import numpy as np
    import torch
    from brats.data import (check_cache, load_manifest, preprocess_case,
                            save_prediction, source_signature)
    from brats.dataset import BraTSSlices

    manifest = load_manifest(args.manifest)
    case_id = manifest["splits"]["train"][0]
    record = manifest["cases"][case_id]
    original_signature = source_signature(args.data_root, record)
    cache = args.output_dir / "sample_cache"
    status = preprocess_case(args.data_root, cache, record)
    metadata = check_cache(cache, record, args.data_root)
    image = np.load(cache / case_id / "image.npy", mmap_mode="r")
    labels = np.load(cache / case_id / "label.npy", mmap_mode="r")
    if not np.isfinite(image).all() or not np.isin(labels, [0, 1, 2, 4]).all():
        raise RuntimeError("Invalid real-sample image or labels")

    # A temporary in-memory view only; never overwrite the fixed patient split.
    subset = {"splits": {"train": [case_id]}, "cases": {case_id: record}}
    dataset = BraTSSlices(subset, cache, input_size=256, augment=False)
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in indices:
        item = dataset[index]
        if (tuple(item["image"].shape) != (4, 256, 256)
                or tuple(item["label"].shape) != (256, 256)
                or not torch.isfinite(item["image"]).all()
                or not torch.all((item["label"] >= 0) & (item["label"] <= 3))):
            raise RuntimeError(f"Invalid padded sample slice {index}")
    np.random.seed(0)
    augmented = BraTSSlices(subset, cache, input_size=256, augment=True)[indices[len(indices) // 2]]
    if not torch.isfinite(augmented["image"]).all():
        raise RuntimeError("Non-finite augmented sample")
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)))
    if tuple(batch["image"].shape) != (min(2, len(dataset)), 4, 256, 256):
        raise RuntimeError("Invalid CPU DataLoader batch")

    # Round-trip the ground truth, not a model prediction.
    output = args.output_dir / "sample_label_roundtrip.nii.gz"
    original_labels = labels.transpose(1, 2, 0)
    save_prediction(original_labels, metadata, output)
    restored = nib.load(output)
    if (not np.array_equal(np.asanyarray(restored.dataobj), original_labels)
            or not np.allclose(restored.affine, metadata["affine"])):
        raise RuntimeError("Real-sample label/geometry round-trip failed")
    if source_signature(args.data_root, record) != original_signature:
        raise RuntimeError("Raw sample files changed during the check")
    return {"case_id": case_id, "manifest_sha256": manifest["sha256"], "cache_status": status,
            "image_shape_zcxy": list(image.shape), "raw_labels": np.unique(labels).tolist(),
            "checked_slices": indices, "batch_shape": list(batch["image"].shape),
            "label_roundtrip": str(output.resolve()), "source_signature_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Preprocess ONE training patient and check slices")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "datasets/BraTS2020")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/brats2020/split_seed42.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local_checks")
    args = parser.parse_args()
    import torch
    from brats.data import write_json

    torch.set_num_threads(2)
    report = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "torch_cuda_build": torch.version.cuda,
              "packages": {p: version(p) for p in ("torch", "torchvision", "timm", "einops",
                           "numpy", "scipy", "nibabel", "pytest", "matplotlib", "pillow", "h5py")},
              "ordinary_modules": check_modules()}
    if args.sample:
        report["sample"] = check_sample(args)
    report["status"] = "passed"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    print(f"Local CPU checks passed. Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
