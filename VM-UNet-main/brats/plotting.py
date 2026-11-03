"""CPU-only offline figures with exported source tables and provenance.

Training deliberately never imports this module. Case views use stored voxel
axes (not a radiological left/right convention) without reorientation/resampling.
"""
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from .data import MODALITIES, check_cache, load_manifest, normalize_volume, source_signature, write_json

REGIONS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}
LABEL_NAMES = {
    1: "Necrotic and non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    4: "Enhancing tumor (ET)",
}
PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}


def configure(path, dpi=None, formats=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import is_color_like
    style = json.loads(Path(path).read_text(encoding="utf-8"))
    if dpi is not None:
        style["dpi"] = dpi
    if formats is not None:
        style["formats"] = formats
    if not isinstance(style["dpi"], int) or style["dpi"] < 50:
        raise ValueError("dpi must be an integer >= 50")
    if not style["formats"] or not set(style["formats"]) <= {"png", "pdf", "svg"}:
        raise ValueError("formats must contain png, pdf and/or svg")
    if not 0 < style["overlay_alpha"] <= 1:
        raise ValueError("overlay_alpha must be in (0, 1]")
    if any(not is_color_like(style["label_colors"][str(v)]) for v in (1, 2, 4)):
        raise ValueError("Invalid label color")
    matplotlib.rcParams.update({"font.family": style["font_family"], "font.size": 10,
                               "pdf.fonttype": 42, "svg.fonttype": "none"})
    return style


def save_figure(fig, output, name, style):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for extension in dict.fromkeys(style["formats"]):
            fig.savefig(output / f"{name}.{extension}", dpi=style["dpi"], bbox_inches="tight")
    finally:
        plt.close(fig)


def write_table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def data_context(args):
    manifest = load_manifest(args.manifest)
    if manifest["purpose"] != "training":
        raise ValueError("These figures require a labeled training manifest")
    return manifest, {"manifest": str(args.manifest.resolve()), "manifest_sha256": manifest["sha256"],
                      "split": args.split, "data_source": "cache" if args.cache_dir else "raw",
                      "source_root": str((args.cache_dir or args.data_root).resolve())}


def load_label(record, data_root, cache_dir=None):
    """Return original-label XYZ data and physical geometry; no MRI intensities."""
    if cache_dir is not None:
        meta = check_cache(cache_dir, record)
        labels = np.load(Path(cache_dir) / record["id"] / "label.npy", mmap_mode="r").transpose(1, 2, 0)
    else:
        import nibabel as nib
        reference = nib.load(str(Path(data_root) / record["images"]["t1"]))
        image = nib.load(str(Path(data_root) / record["label"]))
        if (len(reference.shape) != 3 or image.shape != reference.shape or
                not np.isfinite(reference.affine).all() or
                not np.allclose(image.affine, reference.affine, atol=1e-4)):
            raise ValueError(f"Invalid or unaligned label geometry: {record['id']}")
        units = reference.header.get_xyzt_units()[0]
        scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
        if scale is None:
            raise ValueError(f"Unknown spatial units for {record['id']}")
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0) * scale
        if not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Invalid voxel spacing")
        meta = {"shape_xyz": list(reference.shape), "spacing_xyz_mm": spacing.tolist(),
                "affine": reference.affine.tolist(), "source_signature": source_signature(data_root, record)}
        labels = np.asanyarray(image.dataobj)
    # Validate before any integer conversion, including NaN/fractional labels.
    if not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError(f"Invalid BraTS labels: {record['id']}")
    return labels, meta


def load_case(record, data_root, cache_dir=None):
    labels, meta = load_label(record, data_root, cache_dir)
    if cache_dir is not None:
        images = np.load(Path(cache_dir) / record["id"] / "image.npy", mmap_mode="r").transpose(1, 2, 3, 0)
        if not np.isfinite(images).all():
            raise ValueError("Nonfinite cached MRI")
    else:
        import nibabel as nib
        images = []
        for modality in MODALITIES:
            image = nib.load(str(Path(data_root) / record["images"][modality]))
            if image.shape != labels.shape or not np.allclose(image.affine, meta["affine"], atol=1e-4):
                raise ValueError(f"Unaligned MRI: {record['id']}/{modality}")
            # Match the production float16 cache, including its quantization.
            images.append(normalize_volume(image.get_fdata(dtype=np.float32)).astype(np.float16))
        images = np.stack(images)
        if not np.isfinite(images).all():
            raise ValueError("Normalization overflow")
    return images, labels, meta


def select_slices(labels, plane, region="WT", indices=None, radius=0):
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    axis = PLANES[plane]
    if indices is None:
        mask = np.isin(labels, REGIONS[region])
        areas = mask.sum(axis=tuple(i for i in range(3) if i != axis))
        indices = [int(areas.argmax()) if areas.max() else labels.shape[axis] // 2]
    if any(i < 0 or i >= labels.shape[axis] for i in indices):
        raise ValueError(f"{plane} index outside [0, {labels.shape[axis] - 1}]")
    return sorted({j for i in indices for j in range(max(0, i - radius), min(labels.shape[axis], i + radius + 1))})


def slice_view(volume, plane, index, crop=None):
    view = np.take(volume, index, axis=PLANES[plane]).T
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= view.shape[1] and 0 <= y0 < y1 <= view.shape[0]):
            raise ValueError(f"Crop exceeds displayed pixel bounds {view.shape[::-1]}")
        view = view[y0:y1, x0:x1]
    return view


def plot_case(args, style):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    if args.plane == "all" and (args.slices is not None or args.crop is not None):
        raise ValueError("--slices/--crop requires a single --plane")
    manifest, context = data_context(args)
    ids = manifest["splits"][args.split]
    case_id = args.case_id or ids[0]
    if case_id not in ids:
        raise ValueError(f"{case_id} does not belong to selected split {args.split}")
    images, labels, meta = load_case(manifest["cases"][case_id], args.data_root, args.cache_dir)
    output = args.output_dir / case_id
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")
    planes = list(PLANES) if args.plane == "all" else [args.plane]
    exported, rows = {}, []
    for plane in planes:
        indices = select_slices(labels, plane, args.region, args.slices, args.radius)
        remaining = [i for i in range(3) if i != PLANES[plane]]
        aspect = meta["spacing_xyz_mm"][remaining[1]] / meta["spacing_xyz_mm"][remaining[0]]
        # Each slice is its own figure; adjacent slices never create a huge canvas.
        for index in indices:
            image_views = [slice_view(images[c], plane, index, args.crop).astype(np.float32) for c in range(4)]
            target = slice_view(labels, plane, index, args.crop)
            overlay = np.zeros(target.shape + (4,))
            for label in (1, 2, 4):
                overlay[target == label] = to_rgba(style["label_colors"][str(label)], style["overlay_alpha"])
            height = max(2.5, 3.05 * target.shape[0] / target.shape[1] * aspect + 0.8)
            fig, axes = plt.subplots(1, 5, figsize=(16, height), constrained_layout=True)
            for ax, image, title in zip(axes, image_views + [image_views[3]],
                                        ["T1", "T1ce", "T2", "FLAIR", "FLAIR + ground truth"]):
                nonzero = image[image != 0]
                lo, hi = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)
                ax.imshow(np.ma.masked_where(image == 0, image), cmap=gray, origin="lower",
                          vmin=lo, vmax=hi, aspect=aspect, interpolation="nearest")
                ax.set_title(title)
                ax.set_axis_off()
            axes[-1].imshow(overlay, origin="lower", aspect=aspect, interpolation="nearest")
            fig.legend(handles=[Patch(color=style["label_colors"][str(v)], label=LABEL_NAMES[v])
                                for v in (1, 2, 4)], loc="upper center", bbox_to_anchor=(0.5, 0),
                       ncol=3, fontsize=9, frameon=False)
            key = f"{plane}_{index:03d}"
            fig.suptitle(f"{case_id} | {args.split} | {plane}, stored index={index}")
            save_figure(fig, output, key, style)
            exported[key + "_images"] = np.stack(image_views)
            exported[key + "_labels"] = target.astype(np.uint8)
            rows.append({"case_id": case_id, "plane": plane, "slice_index": index,
                         **{f"label_{v}_pixels": int((target == v).sum()) for v in (0, 1, 2, 4)}})
    np.savez_compressed(output / "case_slices.npz", **exported)
    write_table(output / "slices.csv", rows)
    write_json(output / "case_metadata.json", {**context, "case_id": case_id, "geometry": meta,
               "selection": "manual" if args.slices is not None else f"maximum {args.region} area; center if absent",
               "orientation": "stored XYZ axes, transpose for display, origin=lower; no reorientation/resampling",
               "normalization": "per-patient per-modality nonzero z-score; float16 quantization",
               "display_window": "1st..99th nonzero percentile per displayed modality slice; zero background rendered black",
               "crop_display_xyxy": args.crop, "radius": args.radius, "slices": rows,
               "label_names": LABEL_NAMES, "style": style})


def patient_statistics(case_id, labels, meta):
    values, counts = np.unique(labels, return_counts=True)
    count = {v: 0 for v in (0, 1, 2, 4)}
    count.update({int(v): int(n) for v, n in zip(values, counts)})
    row = {"case_id": case_id, "total_voxels": int(labels.size), "n_axial_slices": labels.shape[2],
           **{f"label_{v}_voxels": count[v] for v in count}}
    slice_rows = []
    per_region = {}
    # The determinant also handles voxel volume for a non-orthogonal affine.
    affine_volume = abs(float(np.linalg.det(np.asarray(meta["affine"])[:3, :3])))
    # Cached affine may use meter/micron units; spacing is always in millimeters.
    native_lengths = np.linalg.norm(np.asarray(meta["affine"])[:3, :3], axis=0)
    mm_scale = np.asarray(meta["spacing_xyz_mm"]) / native_lengths
    voxel_ml = affine_volume * float(np.prod(mm_scale)) / 1000
    for region, region_labels in REGIONS.items():
        per_region[region] = np.isin(labels, region_labels).sum(axis=(0, 1))
        n = sum(count[v] for v in region_labels)
        row.update({f"{region}_voxels": n, f"{region}_ml": n * voxel_ml,
                    f"{region}_positive_slices": int((per_region[region] > 0).sum())})
    for z in range(labels.shape[2]):
        slice_rows.append({"case_id": case_id, "slice_index": z,
                           **{f"{r}_voxels": int(per_region[r][z]) for r in REGIONS}})
    return row, slice_rows


def plot_stats(args, style):
    import matplotlib.pyplot as plt
    manifest, context = data_context(args)
    ids = manifest["splits"][args.split]
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    selected = ids[:args.limit] if args.limit is not None else ids
    rows, slices, signatures = [], [], {}
    for i, case_id in enumerate(selected, 1):
        labels, meta = load_label(manifest["cases"][case_id], args.data_root, args.cache_dir)
        row, patient_slices = patient_statistics(case_id, labels, meta)
        rows.append(row)
        signatures[case_id] = meta.get("source_signature")
        slices.extend(patient_slices)
        if i == 1 or i % 25 == 0 or i == len(selected):
            print(f"[stats {i}/{len(selected)}] {case_id}", flush=True)
    output = args.output_dir
    partial = len(selected) != len(ids)
    title = f"{args.split}, n={len(selected)}" + (" (PARTIAL PREVIEW)" if partial else "")
    totals = {v: sum(row[f"label_{v}_voxels"] for row in rows) for v in (0, 1, 2, 4)}
    denominator = sum(totals.values())
    proportions = [{"label": v, "voxels": totals[v], "percent_whole_grid": 100 * totals[v] / denominator,
                    "percent_tumor_only": 100 * totals[v] / max(1, denominator - totals[0]) if v else ""}
                   for v in totals]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, denom, caption in ((axes[0], [0, 1, 2, 4], denominator, "All voxels (includes outside-brain background)"),
                                        (axes[1], [1, 2, 4], denominator - totals[0], "Tumor voxels only")):
        percentages = [100 * totals[v] / max(1, denom) for v in values]
        ax.bar([str(v) for v in values], percentages,
               color=[style["label_colors"].get(str(v), "#999999") for v in values])
        ax.set(xlabel="Exclusive raw label", ylabel="Voxel share (%)", title=caption)
        ax.set_ylim(0, max(1, max(percentages)) * 1.18)
        for i, value in enumerate(percentages):
            ax.annotate(f"{value:.3f}%", (i, value), ha="center", va="bottom")
    fig.suptitle(title)
    save_figure(fig, output, "class_balance", style)
    for name, suffix, ylabel in (("region_volumes", "ml", "Ground-truth region volume (mL)"),
                                  ("positive_slices", "positive_slices", "Tumor-positive axial slices per patient")):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.boxplot([[r[f"{region}_{suffix}"] for r in rows] for region in REGIONS])
        ax.set_xticks([1, 2, 3], list(REGIONS))
        ax.set(ylabel=ylabel, title=title, ylim=(0, None))
        save_figure(fig, output, name, style)
    # Same raw index is only pooled when depths agree; otherwise avoid implying registration.
    depths = {r["n_axial_slices"] for r in rows}
    axial_rows = []
    if len(depths) == 1:
        depth = next(iter(depths))
        for z in range(depth):
            current = [s for s in slices if s["slice_index"] == z]
            axial_rows.append({"slice_index": z, "n_patients": len(rows),
                               **{f"{r}_positive_patient_fraction": sum(s[f"{r}_voxels"] > 0 for s in current) / len(rows)
                                  for r in REGIONS}})
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for region in REGIONS:
            ax.plot([r["slice_index"] for r in axial_rows],
                    [r[f"{region}_positive_patient_fraction"] for r in axial_rows], label=region)
        ax.set(xlabel="Stored axial index (not anatomical registration)", ylabel="Fraction of patients with region",
               ylim=(0, 1), title=title)
        ax.legend()
        save_figure(fig, output, "axial_presence", style)
        write_table(output / "axial_presence.csv", axial_rows)
    else:
        warnings.warn("Different depths: pooled raw-index axial presence plot omitted")
    write_table(output / "class_balance.csv", proportions)
    write_table(output / "patient_statistics.csv", rows)
    write_table(output / "slice_statistics.csv", slices)
    write_json(output / "stats_metadata.json", {**context, "n_patients": len(rows), "partial": partial,
               "case_ids": selected, "style": style, "regions": REGIONS,
               "source_signatures_size_mtime": signatures,
               "denominator": "entire stored image grid, including outside-brain background",
               "volume_unit": "mL; determinant of affine converted to mm^3 / 1000",
               "axial_presence_available": len(depths) == 1})


def read_history(work_dir):
    """Read a snapshot safely while training appends; never invent missing data."""
    path = Path(work_dir) / "history.jsonl"
    raw = path.read_bytes()
    lines = raw.splitlines()
    records = []
    previous = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not raw.endswith(b"\n"):
                warnings.warn(f"Ignoring an unfinished last history line: {path}")
                break
            raise ValueError(f"Malformed history at {path}:{i + 1}")
        epoch = row["epoch"]
        if not isinstance(epoch, int) or epoch <= previous:
            raise ValueError(f"History epochs must be strictly increasing: {path}")
        previous = epoch
        for key in ("train_loss", "train_ce_loss", "train_dice_loss", "learning_rate", "train_seconds", "val_mean_dice"):
            if key in row and not np.isfinite(row[key]):
                raise ValueError(f"Nonfinite {key} at {path}:{i + 1}")
        if "val_mean_dice" in row and "val_dice" not in row:
            summary_path = Path(work_dir) / "validation" / f"epoch_{epoch:03d}" / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["val_dice"] = {r: summary["regions"][r]["dice"]["mean"] for r in REGIONS}
        for region, score in row.get("val_dice", {}).items():
            if region not in REGIONS or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid validation Dice at {path}:{i + 1}")
        if "val_mean_dice" in row and not 0 <= row["val_mean_dice"] <= 1:
            raise ValueError(f"Invalid validation Mean Dice at {path}:{i + 1}")
        records.append(row)
    if not records:
        raise ValueError(f"No complete training records in {path}")
    return records, hashlib.sha256(raw).hexdigest()


def plot_curves(args, style):
    import matplotlib.pyplot as plt
    names = args.names if args.names is not None else [p.name for p in args.work_dirs]
    if len(names) != len(args.work_dirs) or len(set(names)) != len(names):
        raise ValueError("Provide one unique legend name per work directory")
    runs = [(name, path, *read_history(path)) for name, path in zip(names, args.work_dirs)]
    output = args.output_dir
    fields = ["experiment", "epoch", "train_loss", "train_ce_loss", "train_dice_loss", "learning_rate",
              "train_seconds", "cumulative_train_seconds", "val_mean_dice", "val_WT_dice", "val_TC_dice", "val_ET_dice"]
    flat, metadata = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    region_fig, region_axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    component_fig, component_axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    schedule_fig, schedule_axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    has_validation, has_components, has_regions = False, [False, False], [False] * 3
    for name, directory, records, snapshot_sha in runs:
        epochs = [r["epoch"] for r in records]
        losses = [r.get("train_loss", np.nan) for r in records]
        line, = axes[0].plot(epochs, losses, label=name)
        color = line.get_color()
        valid = [r for r in records if "val_mean_dice" in r]
        best = max(valid, key=lambda r: r["val_mean_dice"]) if valid else None
        if valid:
            has_validation = True
            axes[1].plot([r["epoch"] for r in valid], [r["val_mean_dice"] for r in valid],
                         marker="o", color=color, label=f"{name} (best e{best['epoch']})")
            axes[1].scatter([best["epoch"]], [best["val_mean_dice"]], marker="*", s=140, color=color)
        if records[-1].get("early_stopping", {}).get("stopped"):
            for ax in axes:
                ax.axvline(epochs[-1], color=color, linestyle=":", alpha=0.5)
        for i, region in enumerate(REGIONS):
            samples = [r for r in records if region in r.get("val_dice", {})]
            if samples:
                has_regions[i] = True
                region_axes[i].plot([r["epoch"] for r in samples], [r["val_dice"][region] for r in samples],
                                    marker="o", label=name, color=color)
        for i, key in enumerate(("train_ce_loss", "train_dice_loss")):
            if any(key in r for r in records):
                has_components[i] = True
                # NaN gaps keep missing legacy epochs visible; no zero filling.
                component_axes[i].plot(epochs, [r.get(key, np.nan) for r in records], label=name, color=color)
        schedule_axes[0].plot(epochs, [r.get("learning_rate", np.nan) for r in records], label=name, color=color)
        elapsed, complete_time = 0.0, True
        for record in records:
            complete_time = complete_time and "train_seconds" in record
            elapsed += record.get("train_seconds", 0)
            flat.append({"experiment": name, **{k: record.get(k, "") for k in fields[1:]
                                                 if not k.startswith("val_")},
                         "cumulative_train_seconds": elapsed if complete_time else "",
                         "val_mean_dice": record.get("val_mean_dice", ""),
                         **{f"val_{r}_dice": record.get("val_dice", {}).get(r, "") for r in REGIONS}})
        # Includes epoch training time only; validation/checkpoint time is excluded.
        times = np.cumsum([r.get("train_seconds", 0) for r in records])
        if valid and all("train_seconds" in r for r in records):
            schedule_axes[1].plot([times[records.index(r)] / 3600 for r in valid],
                                  [r["val_mean_dice"] for r in valid], marker="o", label=name, color=color)
        run_path = directory / "run.json"
        run_info = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
        metadata.append({"name": name, "work_dir": str(directory.resolve()), "history_snapshot_sha256": snapshot_sha,
                         "epochs": epochs, "best_observed_epoch": best["epoch"] if best else None,
                         "early_stopped": records[-1].get("early_stopping", {}).get("stopped", False),
                         "manifest_sha256": run_info.get("manifest_sha256"),
                         "missing_components": [k for k in ("train_ce_loss", "train_dice_loss")
                                                if not any(k in r for r in records)]})
    for ax, title in zip(axes, ("Training total loss", "Validation Mean Dice")):
        ax.set(xlabel="Epoch", ylabel="Loss" if ax is axes[0] else "Dice (0..1)", title=title)
        if ax.lines:
            ax.legend()
    if not has_validation:
        axes[1].text(0.5, 0.5, "No validation observations yet", transform=axes[1].transAxes, ha="center")
    save_figure(fig, output, "training_curves", style)
    for i, (ax, region) in enumerate(zip(region_axes, REGIONS)):
        ax.set(xlabel="Epoch", ylabel="Validation Dice (0..1)", title=region)
        if has_regions[i]:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Not recorded", transform=ax.transAxes, ha="center")
    save_figure(region_fig, output, "validation_regions", style)
    for i, (ax, title) in enumerate(zip(component_axes, ("Cross entropy", "WT/TC/ET soft Dice loss"))):
        ax.set(xlabel="Epoch", ylabel="Training loss", title=title)
        if has_components[i]:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Not recorded in legacy logs", transform=ax.transAxes, ha="center")
    save_figure(component_fig, output, "loss_components", style)
    schedule_axes[0].set(xlabel="Epoch", ylabel="Learning rate")
    schedule_axes[1].set(xlabel="Cumulative training hours (excludes validation / checkpoint I/O)",
                         ylabel="Validation Mean Dice")
    for ax in schedule_axes:
        if ax.lines:
            ax.legend()
    save_figure(schedule_fig, output, "schedule_and_time", style)
    checksums = {m["manifest_sha256"] for m in metadata if m["manifest_sha256"]}
    if len(checksums) > 1:
        warnings.warn("Compared runs have different patient split checksums")
    write_table(output / "curve_data.csv", flat, fields)
    write_json(output / "curves_metadata.json", {"runs": metadata, "style": style,
               "validation": "Only observed validation epochs; no smoothing/interpolation of missing observations",
               "time_scope": "sum of logged train_seconds; excludes validation/checkpoint and restart downtime",
               "best_marker": "highest observed validation Mean Dice; no checkpoint loaded",
               "early_stop_marker": "dotted line only when the final history record says stopped"})
