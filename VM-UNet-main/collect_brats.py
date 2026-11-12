"""Collect one training run's validation results into research/experiments/<EXP>/.

Numbers in experiment reports come from these generated files, never from hand
copies. The held-out test set is exported only with --include-test, into
<EXP>/final/, and never enters summary.json or index.md.
"""
import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
DEFAULT_OUT = SOURCE.parent / "research" / "experiments"
REGIONS = ("WT", "TC", "ET")
EXTENSIONS = ("shallow_refinement", "detail_fusion")
EXP_PATTERN = re.compile(r"EXP-\d{3}")
INDEX_HEADER = "<!-- 由 collect_brats.py 生成，勿手改 -->"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_history(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_epoch(history, epoch=None):
    if epoch is not None:
        return epoch, "fixed"
    if history and history[-1].get("best_epoch") is not None:
        return history[-1]["best_epoch"], "best"
    validated = [row for row in history if row.get("val_mean_dice") is not None]
    if not validated:
        raise ValueError("history.jsonl contains no validation results")
    return max(validated, key=lambda row: row["val_mean_dice"])["epoch"], "best"


def empty_et_statistics(per_patient_csv):
    with Path(per_patient_csv).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    empty = [row for row in rows if int(row["ET_target_voxels"]) == 0]
    false_positive = sum(int(row["ET_prediction_voxels"]) > 0 for row in empty)
    nonempty = [float(row["ET_dice"]) for row in rows if int(row["ET_target_voxels"]) > 0]
    return f"{false_positive}/{len(empty)}", sum(nonempty) / len(nonempty) if nonempty else None


def config_summary(config):
    pretrained = config.get("pretrained")
    warm_start = config.get("warm_start")
    return {"epochs": config.get("epochs"), "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "pretrained": Path(pretrained).name if pretrained else None,
            "warm_start": warm_start.get("checkpoint") if isinstance(warm_start, dict) else None,
            "extension": {name: config[name] for name in EXTENSIONS if name in config}}


def default_test_dir(run_dir):
    candidates = sorted((path for path in run_dir.glob("test_*/evaluation_test") if path.is_dir()),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No test_*/evaluation_test directory in {run_dir}")
    return candidates[-1]


def collect(run_dir, exp, title, out, epoch=None, include_test=False, test_dir=None, force=False):
    run_dir, out = Path(run_dir).resolve(), Path(out)
    if not EXP_PATTERN.fullmatch(exp):
        raise ValueError("--exp must look like EXP-001")
    history = read_history(run_dir / "history.jsonl")
    selected, selection = select_epoch(history, epoch)
    validation = run_dir / "validation" / f"epoch_{selected:03d}"
    if not (validation / "summary.json").is_file() or not (validation / "per_patient.csv").is_file():
        raise FileNotFoundError(f"Missing validation results for epoch {selected}: {validation}")
    copies = {"config.json": run_dir / "config.json", "history.jsonl": run_dir / "history.jsonl",
              "val_per_patient.csv": validation / "per_patient.csv"}
    if include_test:
        test_dir = Path(test_dir).resolve() if test_dir else default_test_dir(run_dir)
        copies["final/test_summary.json"] = test_dir / "summary.json"
        copies["final/test_per_patient.csv"] = test_dir / "per_patient.csv"
    target = out / exp
    outputs = ["summary.json", *copies]
    existing = [name for name in outputs if (target / name).exists()]
    if existing and not force:
        raise FileExistsError(f"{target} already has {', '.join(existing)}; pass --force to overwrite")
    for source in copies.values():
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}")

    config = read_json(run_dir / "config.json")
    run_info = read_json(run_dir / "run.json") if (run_dir / "run.json").is_file() else {}
    metrics = read_json(validation / "summary.json")
    empty_fp, nonempty_et = empty_et_statistics(validation / "per_patient.csv")
    val = {"mean_dice": metrics["mean_dice"], "mean_hd95_mm": metrics["mean_hd95_mm"]}
    for region in REGIONS:
        values = metrics["regions"][region]
        val[region] = {"dice": values["dice"]["mean"], "hd95_mm": values["hd95_mm"]["mean"]}
    val.update({"et_empty_false_positive": empty_fp, "et_nonempty_dice": nonempty_et})
    summary = {
        "exp": exp, "title": title, "run_dir": run_dir.name,
        "collected_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "git": run_info.get("git"), "config": config_summary(config),
        "epochs_completed": max((row["epoch"] for row in history), default=0),
        "train_hours": round(sum(row.get("train_seconds", 0) for row in history) / 3600, 2),
        "parameters": run_info.get("parameters"),
        "selected_epoch": selected, "selection": selection, "val": val}

    (target / "final" if include_test else target).mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        shutil.copyfile(source, target / name)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    write_index(out)
    return summary


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def write_index(out):
    summaries = sorted((read_json(path) for path in Path(out).glob("EXP-*/summary.json")),
                       key=lambda item: item["exp"])
    reference = next((item for item in summaries if item["exp"] == "EXP-001"), None)
    lines = [INDEX_HEADER, "# 实验总表（验证集）", "",
             "| EXP | 标题 | 选定/完成轮次 | Mean Dice | WT | TC | ET | Mean HD95 | 空ET假阳性 | ΔMean vs EXP-001 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        val = item["val"]
        delta = ("—" if reference is None or item is reference else
                 f"{100 * (val['mean_dice'] - reference['val']['mean_dice']):+.2f}")
        lines.append(f"| [{item['exp']}]({item['exp']}/report.md) | {item['title']} | "
                     f"{item['selected_epoch']}/{item['epochs_completed']} | {percent(val['mean_dice'])} | "
                     + " | ".join(percent(val[region]["dice"]) for region in REGIONS)
                     + f" | {val['mean_hd95_mm']:.2f} | {val['et_empty_false_positive']} | {delta} |")
    lines += ["", "Dice 为百分数，HD95 单位 mm；Δ 为百分点。数值来自各目录的 `summary.json`，测试集结果不进入本表。"]
    (Path(out) / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory, e.g. results/<run>")
    parser.add_argument("--exp", required=True, help="Experiment ID, e.g. EXP-002")
    parser.add_argument("--title", required=True, help="Short experiment title shown in index.md")
    parser.add_argument("--epoch", type=int, help="Validated epoch to collect (default: the run's best epoch)")
    parser.add_argument("--include-test", action="store_true",
                        help="Also copy sealed test results into <EXP>/final/ (FINAL tasks only)")
    parser.add_argument("--test-dir", type=Path,
                        help="evaluation_test directory (default: newest <run>/test_*/evaluation_test)")
    parser.add_argument("--force", action="store_true", help="Overwrite previously collected files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Experiments directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    if args.test_dir and not args.include_test:
        parser.error("--test-dir requires --include-test")
    run_dir = args.run_dir if args.run_dir.is_absolute() or args.run_dir.exists() else SOURCE / args.run_dir
    try:
        summary = collect(run_dir, args.exp, args.title, args.out, args.epoch,
                          args.include_test, args.test_dir, args.force)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Index:", Path(args.out) / "index.md")


if __name__ == "__main__":
    sys.exit(main())
