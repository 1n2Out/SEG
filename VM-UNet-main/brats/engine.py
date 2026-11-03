import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

from .data import check_cache, save_prediction, write_json
from .dataset import padding
from .metrics import METRICS, REGIONS, evaluate_regions, summarize


@torch.no_grad()
def predict_volume(model, images, device, input_size=256, batch_size=8, amp=False):
    """Z,C,X,Y -> original-label X,Y,Z. Includes every axial slice."""
    model.eval()
    z_count, channels, x, y = images.shape
    if channels != 4 or batch_size < 1:
        raise ValueError("Expected four modalities and positive inference batch size")
    pads = padding((x, y), input_size)
    prediction = np.empty((z_count, x, y), dtype=np.uint8)
    for start in range(0, z_count, batch_size):
        batch = np.array(images[start:start + batch_size], dtype=np.float32)
        batch = np.pad(batch, ((0, 0), (0, 0)) + pads)
        batch = torch.from_numpy(batch).to(device)
        with autocast(enabled=amp):
            logits = model(batch)
        if logits.shape != (len(batch), 4, input_size, input_size) or not torch.isfinite(logits).all():
            raise ValueError("Invalid BraTS output shape or nonfinite logits")
        logits = logits[:, :, pads[0][0]:pads[0][0] + x, pads[1][0]:pads[1][0] + y]
        labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        labels[labels == 3] = 4
        prediction[start:start + len(labels)] = labels
    return prediction.transpose(1, 2, 0)


def evaluate(model, manifest, cache_dir, split, device, config,
             output_dir, save_predictions=False, limit=None):
    if split not in ("val", "test", "official_val"):
        raise ValueError("Evaluation split must be val, test, or official_val")
    ids = manifest["splits"][split]
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise ValueError(f"No patients in {split}")
    if split == "official_val" and not save_predictions:
        raise ValueError("Official validation has no labels: use prediction export")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, timings = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i, case_id in enumerate(ids, 1):
        record = manifest["cases"][case_id]
        metadata = check_cache(cache_dir, record)
        directory = Path(cache_dir) / case_id
        images = np.load(directory / "image.npy", mmap_mode="r")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = predict_volume(model, images, device, config["input_size"],
                                    config["inference_batch_size"], config["amp"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        if save_predictions:
            save_prediction(prediction, metadata, output_dir / "predictions" / (case_id + ".nii.gz"))
        if split != "official_val":
            target = np.load(directory / "label.npy", mmap_mode="r").transpose(1, 2, 0)
            scores = evaluate_regions(prediction, target, metadata["spacing_xyz_mm"])
            for region, labels in REGIONS.items():
                scores[region]["target_voxels"] = int(np.isin(target, labels).sum())
                scores[region]["prediction_voxels"] = int(np.isin(prediction, labels).sum())
            rows.append({"case_id": case_id, **scores})
            print(f"[{split} {i}/{len(ids)}] {case_id}: " +
                  " ".join(f"{r}={scores[r]['dice']:.4f}" for r in REGIONS), flush=True)
        else:
            print(f"[predict {i}/{len(ids)}] {case_id}", flush=True)
    summary = summarize(rows) if rows else {"n_patients": len(ids), "metrics_available": False}
    summary.update({"split": split, "manifest_sha256": manifest["sha256"],
                    "partial_evaluation": limit is not None, "model_parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(device), "amp": config["amp"],
                    "inference_seconds_per_patient": timings,
                    "timing_scope": "cached array access, padding, transfers, forward, argmax and reconstruction; excludes metrics/NIfTI I/O; no warmup",
                    "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None})
    write_json(output_dir / "summary.json", summary)
    if rows:
        write_json(output_dir / "per_patient.json", rows)
        fields = ["case_id"] + [f"{r}_{m}" for r in REGIONS for m in (*METRICS, "target_voxels", "prediction_voxels")]
        with (output_dir / "per_patient.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"case_id": row["case_id"],
                                 **{f"{r}_{m}": row[r][m] for r in REGIONS for m in (*METRICS, "target_voxels", "prediction_voxels")}})
    return summary
