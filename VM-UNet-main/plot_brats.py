"""Offline BraTS figures: case / stats / curves. No model or torch import."""
import argparse
from pathlib import Path

SOURCE = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("case", "stats", "curves"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--plot-config", type=Path, default=SOURCE / "configs/plotting.json")
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
        sub.add_argument("--dpi", type=int)
        if command != "curves":
            sub.add_argument("--manifest", type=Path, default=SOURCE / "data/brats2020/split_seed42.json")
            sub.add_argument("--data-root", type=Path, default=SOURCE.parent / "datasets/BraTS2020")
            sub.add_argument("--cache-dir", type=Path, help="Read existing v2 cache instead of raw NIfTI")
            sub.add_argument("--split", choices=("train", "val", "test"), default="train")
        if command == "case":
            sub.add_argument("--case-id", help="Default: first patient in the selected split")
            sub.add_argument("--plane", choices=("axial", "coronal", "sagittal", "all"), default="all")
            sub.add_argument("--slices", type=int, nargs="+", help="Zero-based stored voxel indices; requires one plane")
            sub.add_argument("--region", choices=("WT", "TC", "ET"), default="WT", help="Region for automatic max-area slice")
            sub.add_argument("--radius", type=int, default=0, help="Include this many adjacent slices on each side")
            sub.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                             help="Display pixel bounds, applied after transpose; requires one plane")
        elif command == "stats":
            sub.add_argument("--limit", type=int, help="Preview only; output is explicitly marked partial")
        else:
            sub.add_argument("--work-dirs", type=Path, nargs="+", required=True)
            sub.add_argument("--names", nargs="+", help="One legend name per work directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from brats.plotting import configure, plot_case, plot_curves, plot_stats
    style = configure(args.plot_config, args.dpi, args.formats)
    if args.output_dir is None:
        args.output_dir = SOURCE / "results/figures" / args.command
        if args.command == "stats":
            args.output_dir = args.output_dir / args.split
    if args.command == "case":
        plot_case(args, style)
    elif args.command == "stats":
        plot_stats(args, style)
    else:
        plot_curves(args, style)
    print(f"Figures and source data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
