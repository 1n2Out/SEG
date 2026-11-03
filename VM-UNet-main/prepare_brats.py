"""Create a fixed patient split; optionally prepare the disk-backed MRI cache."""
import argparse
from pathlib import Path

from brats.data import load_manifest, make_manifest, make_prediction_manifest, preprocess_case, write_json

SOURCE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SOURCE.parent / "datasets/BraTS2020")
    parser.add_argument("--manifest", type=Path, help="Defaults to split_seed42.json, or official_val.json with --official-val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--cache", action="store_true", help="Read/validate NIfTI voxels and build numpy cache")
    parser.add_argument("--cache-dir", type=Path, default=SOURCE / "data/brats2020/cache")
    parser.add_argument("--official-val", action="store_true", help="Prepare only the separate, unlabeled official prediction cohort")
    parser.add_argument("--rebuild-cache", action="store_true", help="Explicitly replace existing preprocessed arrays")
    args = parser.parse_args()
    if args.rebuild_cache and not args.cache:
        parser.error("--rebuild-cache requires --cache")
    if args.manifest is None:
        args.manifest = SOURCE / "data/brats2020" / ("official_val.json" if args.official_val else "split_seed42.json")
    manifest = (make_prediction_manifest(args.data_root) if args.official_val else
                make_manifest(args.data_root, args.seed, args.val_fraction, args.test_fraction))
    if args.manifest.exists():
        if load_manifest(args.manifest) != manifest:
            parser.error("Existing manifest differs. Use a new --manifest path to preserve experiment splits.")
    else:
        write_json(args.manifest, manifest)
    print("Manifest:", args.manifest.resolve(), "SHA256:", manifest["sha256"], flush=True)
    print("Patient counts:", {k: len(v) for k, v in manifest["splits"].items()}, flush=True)
    special = [r for r in manifest["cases"].values()
               if "label" in r and not Path(r["label"]).name.startswith(r["id"])]
    for record in special:
        print("Explicit label alias:", record["id"], "->", record["label"], flush=True)
    if args.cache:
        ids = [case_id for ids in manifest["splits"].values() for case_id in ids]
        for i, case_id in enumerate(ids, 1):
            status = preprocess_case(args.data_root, args.cache_dir, manifest["cases"][case_id], args.rebuild_cache)
            print(f"[{i}/{len(ids)}] {case_id}: {status}", flush=True)


if __name__ == "__main__":
    main()
