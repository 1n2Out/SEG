"""Launch BraTS from one JSON config. Actual jobs run on Linux servers only."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from brats.provenance import code_changes_since, git_state
from train_brats import (SOURCE, load_training_config, resolve_pretrained_path,
                         resolve_warm_start_path, validate_config)


def source_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Launch paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else SOURCE / path).resolve()


def timestamp_label():
    """Short month/day and minute label in the user's Asia/Shanghai time."""
    return datetime.now(timezone(timedelta(hours=8))).strftime("%m%d_%H%M")


def new_task_paths(prefix, log_dir, log_prefix=""):
    """Plan without writes; an existing output, log or PID is never reused."""
    stem = prefix.name + "_" + timestamp_label()
    index = 0
    while True:
        name = stem if index == 0 else f"{stem}_{index:02d}"
        directory = prefix.with_name(name)
        log_file = log_dir / (log_prefix + name + ".log")
        if not any(path.exists() for path in (directory, log_file, log_file.with_suffix(".pid"))):
            return directory, log_file
        index += 1


def select_run(base_dir, run_dir):
    if run_dir is not None:
        return source_path(str(run_dir))
    # Retain access to the original fixed directory as well as dated runs.
    candidates = [base_dir] if base_dir.is_dir() else []
    pattern = re.compile(re.escape(base_dir.name) +
                         r"_(?:\d{4}_\d{4}|\d{8}_\d{6}_\d{6})(?:_\d+)?")
    if base_dir.parent.is_dir():
        candidates += sorted(path for path in base_dir.parent.iterdir()
                             if path.is_dir() and pattern.fullmatch(path.name))
    if len(candidates) > 1:
        choices = "\n".join(str(path) for path in candidates)
        raise ValueError("Multiple training runs found; select one with --run-dir:\n" + choices)
    return candidates[0] if candidates else base_dir


def parse_gpus(value):
    """CLI form of launch.gpus: comma-separated indices such as "3" or "0,1"."""
    try:
        return [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--gpus takes comma-separated GPU indices, e.g. 3 or 0,1") from None


def make_plan(action, config_path, run_dir=None, gpus=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_config = load_training_config(config_path)
    validate_config(training_config)
    launch = config.get("launch", {})
    if not isinstance(launch, dict):
        raise ValueError("launch must be an object")
    source = "launch.gpus" if gpus is None else "--gpus"
    gpus = launch.get("gpus", [0]) if gpus is None else gpus
    if (not isinstance(gpus, list) or not gpus or
            any(type(gpu) is not int or gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError(source + " must be a nonempty list of distinct nonnegative GPU indices")
    save_predictions = launch.get("save_predictions", True)
    if not isinstance(save_predictions, bool):
        raise ValueError("launch.save_predictions must be a boolean")
    base_dir = source_path(launch.get("work_dir", "results/brats2020_seed42"))
    cache_dir = source_path(launch.get("cache_dir", "data/brats2020/cache"))
    manifest = source_path(launch.get("manifest", "data/brats2020/split_seed42.json"))
    data_root = source_path(launch.get("data_root", "../datasets/BraTS2020"))
    log_dir = source_path(launch.get("log_dir", "logs"))
    if log_dir == base_dir or base_dir in log_dir.parents:
        raise ValueError("launch.log_dir must be outside launch.work_dir")
    if run_dir is not None and action not in ("resume", "test"):
        raise ValueError("--run-dir is only supported for resume and test")
    selected_run = None
    if action == "train":
        work_dir, log_file = new_task_paths(base_dir, log_dir)
        selected_run = work_dir
    elif action == "smoke":
        work_dir, log_file = new_task_paths(base_dir.with_name(base_dir.name + "_smoke"), log_dir)
    elif action in ("resume", "test"):
        selected_run = select_run(base_dir, run_dir)
        if action == "resume":
            work_dir = selected_run
            log_file = log_dir / (selected_run.name + ".log")
        else:
            work_dir, log_file = new_task_paths(selected_run / "test", log_dir, selected_run.name + "_")
    else:  # Cache preparation shares the configured data cache, but isolates logs.
        work_dir = base_dir
        _, log_file = new_task_paths(base_dir.with_name(base_dir.name + "_prepare"), log_dir)
    for directory in (work_dir, selected_run):
        if directory is not None and (log_dir == directory or directory in log_dir.parents):
            raise ValueError("launch.log_dir must be outside the task and training run directories")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in (gpus[:1] if action == "test" else gpus))
    command = [sys.executable, "-u"]
    if action == "prepare":
        command += [str(SOURCE / "prepare_brats.py"), "--cache", "--data-root", str(data_root),
                    "--manifest", str(manifest), "--cache-dir", str(cache_dir)]
    else:
        if len(gpus) > 1 and action != "test":
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                        f"--nproc_per_node={len(gpus)}"]
        mode = {"train": "train", "resume": "train", "smoke": "smoke", "test": "evaluate"}[action]
        command += [str(SOURCE / "train_brats.py"), "--mode", mode, "--config", str(config_path),
                    "--manifest", str(manifest), "--cache-dir", str(cache_dir), "--work-dir", str(work_dir)]
        if action == "resume":
            command += ["--resume", str(selected_run / "latest.pth")]
        elif action == "test":
            command += ["--checkpoint", str(selected_run / "best.pth"), "--split", "test"]
            if save_predictions:
                command.append("--save-predictions")
    return {"action": action, "command": command, "environment": environment,
            "work_dir": work_dir, "run_dir": selected_run, "base_dir": base_dir,
            "data_root": data_root, "cache_dir": cache_dir,
            "manifest": manifest, "log_file": log_file,
            "pretrained_path": resolve_pretrained_path(config.get("pretrained")),
            "warm_start_path": resolve_warm_start_path(training_config.get("warm_start"))}


def check_inputs(plan):
    action, work_dir = plan["action"], plan["work_dir"]
    if action == "prepare":
        if not plan["data_root"].is_dir():
            raise FileNotFoundError(f"Set launch.data_root to your BraTS directory: {plan['data_root']}")
        return
    if not plan["manifest"].is_file():
        raise FileNotFoundError(f"Missing manifest: {plan['manifest']}")
    if not plan["cache_dir"].is_dir():
        raise FileNotFoundError("Missing cache; first run: python run_brats.py prepare")
    pretrained = plan["pretrained_path"]
    if action in ("train", "smoke") and pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    # Warm start initializes new runs only; resume and test use the run's own checkpoints.
    warm_start = plan.get("warm_start_path")
    if action in ("train", "smoke") and warm_start is not None and not warm_start.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start}")
    if action in ("train", "smoke", "test") and work_dir.exists():
        raise FileExistsError("New task directory already exists; launch again to allocate a fresh directory")
    required = ("latest.pth", "best.pth") if action == "resume" else ("best.pth",) if action == "test" else ()
    for name in required:
        if not (plan["run_dir"] / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {plan['run_dir'] / name}")


def provenance_problems(plan, allow_dirty=False, allow_code_change=False, confirm_test=False):
    """Return (errors, warnings) of the source-version and test-set rules for a plan."""
    action, errors, warnings = plan["action"], [], []
    if action not in ("train", "resume", "test"):
        return errors, warnings
    if action == "train" and not allow_dirty:
        # Same scope as resume/test: reports and STATE edits on B must not block launches,
        # while any uncommitted source or config change (tracked or new) does.
        current = git_state(SOURCE)
        changed = None if current is None else code_changes_since(SOURCE, current["commit"])
        if changed is None:
            errors.append("Git state unavailable; formal training must run from a git checkout "
                          "(use --allow-dirty only for exploratory runs)")
        elif changed:
            errors.append("Uncommitted code changes; commit them before formal training or pass --allow-dirty:\n"
                          + "\n".join(changed[:20]))
    if action in ("resume", "test"):
        try:
            recorded = json.loads((plan["run_dir"] / "run.json").read_text(encoding="utf-8")).get("git")
        except (OSError, ValueError):
            recorded = None
        if not recorded or not recorded.get("commit"):
            warnings.append("Training run has no recorded git commit; code equality cannot be checked")
        elif not allow_code_change:
            # Docs and reports change constantly; only source changes since the run matter.
            changed = code_changes_since(SOURCE, recorded["commit"])
            if changed is None:
                errors.append(f"Cannot compare code with the run's commit {recorded['commit']} "
                              "(git unavailable or commit not fetched); fetch it or pass --allow-code-change")
            elif changed:
                listed = changed[:20] + ([f"... and {len(changed) - 20} more"] if len(changed) > 20 else [])
                errors.append(f"Code changed since the run's commit {recorded['commit']}; check out that "
                              "commit or pass --allow-code-change:\n" + "\n".join(listed))
    if action == "test" and not confirm_test:
        errors.append("The held-out test set is sealed; evaluate it only for a user-approved FINAL task "
                      "and pass --confirm-test")
    return errors, warnings


def reserve_output(plan):
    # Empty directories are accepted by train_brats. Exclusive mkdir also guards
    # simultaneous launches which planned the same timestamp before either wrote.
    if plan["action"] in ("train", "smoke", "test"):
        plan["work_dir"].mkdir(parents=True, exist_ok=False)


def start_background(plan):
    """Detach from the terminal and redirect all standard streams, like nohup."""
    log_file = plan["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Resume appends to its original log; every new task requires a fresh log.
    with log_file.open("ab" if plan["action"] == "resume" else "xb") as stream:
        process = subprocess.Popen(plan["command"], cwd=str(SOURCE), env=plan["environment"],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    log_file.with_suffix(".pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    return process.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="New task names use <base>_MMDD_HHMM in Shanghai time, "
                                            "adding _01, _02, etc. if needed. Each test gets a separate "
                                            "dated subdirectory. Older run names remain supported.")
    parser.add_argument("action", nargs="?", choices=("train", "resume", "smoke", "test", "prepare"), default="train")
    parser.add_argument("--config", type=Path, default=SOURCE / "configs/brats2020.json")
    parser.add_argument("--background", action="store_true", help="Detach and save output under launch.log_dir")
    parser.add_argument("--run-dir", type=Path,
                        help="Original training directory for resume/test; required when multiple runs exist")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting a job")
    parser.add_argument("--gpus", type=parse_gpus,
                        help="Comma-separated GPU indices overriding launch.gpus for this launch, e.g. 3 or 0,1")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="train: allow uncommitted changes (exploratory runs only)")
    parser.add_argument("--allow-code-change", action="store_true",
                        help="resume/test: allow source changes since the run's recorded commit")
    parser.add_argument("--confirm-test", action="store_true",
                        help="test: confirm a user-approved FINAL evaluation of the held-out test set")
    args = parser.parse_args()
    try:
        plan = make_plan(args.action, args.config, args.run_dir, args.gpus)
        print(f"CUDA_VISIBLE_DEVICES={plan['environment']['CUDA_VISIBLE_DEVICES']} "
              + shlex.join(plan["command"]), flush=True)
        if plan["action"] != "prepare":
            print("Output directory:", plan["work_dir"], flush=True)
        if plan["run_dir"] is not None:
            print("Training run:", plan["run_dir"], flush=True)
        errors, warnings = provenance_problems(plan, args.allow_dirty, args.allow_code_change,
                                               args.confirm_test)
        for message in warnings + (errors if args.dry_run else []):
            print("Warning:", message, flush=True)
        if args.dry_run:
            if args.background:
                print(f"Background log: {plan['log_file']}")
                print(f"PID file: {plan['log_file'].with_suffix('.pid')}")
            return
        if sys.platform != "linux":
            raise RuntimeError("Run jobs on the Linux server; use --dry-run for local command inspection")
        if errors:
            raise RuntimeError("\n".join(errors))
        check_inputs(plan)
        reserve_output(plan)
        if args.background:
            pid = start_background(plan)
            print(f"Started launcher PID={pid}; inspect the log for job progress.")
            print("Log:", plan["log_file"])
            print("PID file:", plan["log_file"].with_suffix(".pid"))
            print("Follow:", "tail -f " + shlex.quote(str(plan["log_file"])))
        else:
            os.chdir(SOURCE)
            os.execvpe(plan["command"][0], plan["command"], plan["environment"])
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
