"""BraTS entry point: train / evaluate / predict / smoke (no legacy ISIC utilities)."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from datetime import timedelta

from brats.data import load_manifest, write_json
from brats.provenance import git_state
from brats.training import EarlyStopping

SOURCE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "predict", "smoke"), default="train")
    parser.add_argument("--config", type=Path, default=SOURCE / "configs/brats2020.json")
    parser.add_argument("--manifest", type=Path, default=SOURCE / "data/brats2020/split_seed42.json")
    parser.add_argument("--prediction-manifest", type=Path, default=SOURCE / "data/brats2020/official_val.json")
    parser.add_argument("--cache-dir", type=Path, default=SOURCE / "data/brats2020/cache")
    parser.add_argument("--work-dir", type=Path, default=SOURCE / "results/brats2020_seed42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test", help="For --mode evaluate")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0,
                        help=argparse.SUPPRESS)  # Older torchrun launchers
    return parser.parse_args()


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(path):
    import torch
    # Local experiment files contain optimizer/RNG state, not only tensor weights.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.13 compatibility
        return torch.load(path, map_location="cpu")


# Optional model extensions; each is a VMUNet subclass whose new parameters live
# under the attribute of the same name. At most one is configured per run.
EXTENSIONS = ("shallow_refinement", "detail_fusion")
WARM_START_DEFAULTS = {"checkpoint": None, "new_parameter_lr": None, "new_parameters": None}


def innovation_settings(config):
    """Return (name, resolved settings) of the configured model extension, or None."""
    if "region_presence" in config:
        raise ValueError("region_presence has been retired; use the old run's exports/*_inference.pth "
                         "for evaluation, or restore its archived source for resume")
    names = [name for name in EXTENSIONS if name in config]
    if not names:
        return None
    if len(names) > 1:
        raise ValueError("Configure at most one model extension, found: " + ", ".join(names))
    name = names[0]
    if name == "shallow_refinement":
        from brats.innovations.shallow_refinement import resolve_settings
    else:
        from brats.innovations.detail_fusion import resolve_settings
    return name, resolve_settings(config[name])


def extension_prefixes(config):
    """State-dict prefixes of the parameters added by an active model extension."""
    extension = innovation_settings(config)
    if extension is None or extension[1]["mode"] == "off":
        return ()
    return (extension[0] + ".",)


def input_context(config):
    """Neighbouring slices per side that the configured model reads (0: current slice only)."""
    extension = innovation_settings(config)
    if extension is None or extension[0] != "detail_fusion":
        return 0
    return extension[1]["context_slices"]


def new_parameter_prefixes(config):
    """Prefixes of tensors absent from the warm-start source.

    warm_start.new_parameters declares them explicitly (stacking on a source that
    already has the extension); otherwise the whole active extension is new.
    """
    settings = config.get("warm_start")
    if settings is not None and settings.get("new_parameters") is not None:
        return tuple(settings["new_parameters"])
    return extension_prefixes(config)


def warm_start_settings(config):
    """Screening-only initialization of the full model from an earlier run.

    Formal experiments keep config.pretrained. Warm start loads every original
    parameter from a training checkpoint of the same patient split, so new
    zero-residual extensions begin exactly at that model's output.
    """
    if "warm_start" not in config:
        return None
    settings = config["warm_start"]
    if not isinstance(settings, dict):
        raise ValueError("warm_start must be an object")
    unknown = set(settings) - set(WARM_START_DEFAULTS)
    if unknown:
        raise ValueError("Unknown warm_start settings: " + ", ".join(sorted(unknown)))
    resolved = {**WARM_START_DEFAULTS, **settings}
    if not isinstance(resolved["checkpoint"], str) or not resolved["checkpoint"].strip():
        raise ValueError("warm_start.checkpoint must be a nonempty checkpoint path")
    rate = resolved["new_parameter_lr"]
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                             or not math.isfinite(rate) or rate <= 0):
        raise ValueError("warm_start.new_parameter_lr must be null or a positive finite number")
    declared = resolved["new_parameters"]
    if declared is not None:
        allowed = extension_prefixes(config)
        if (not isinstance(declared, list) or len(set(declared)) != len(declared)
                or any(not isinstance(item, str) or not item.endswith(".")
                       or not item.startswith(allowed) for item in declared)):
            raise ValueError("warm_start.new_parameters must be null or a list of distinct state-dict "
                             "prefixes ending in a dot inside the active extension")
    if config.get("pretrained") is not None:
        raise ValueError("warm_start replaces pretrained initialization; set pretrained to null")
    return resolved


def build_model(config):
    extension = innovation_settings(config)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("VM-UNet selective scan requires a configured CUDA training environment")
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # fail early, before training
    from models.vmunet.vmunet import VMUNet
    kwargs = dict(input_channels=4, num_classes=4, depths=config["depths"],
                  depths_decoder=config["depths_decoder"], drop_path_rate=config["drop_path_rate"],
                  load_ckpt_path=config["pretrained"])
    if extension is not None and extension[1]["mode"] != "off":
        name, settings = extension
        if name == "shallow_refinement":
            from brats.innovations.shallow_refinement.model import ShallowRefinementVMUNet
            return ShallowRefinementVMUNet(shallow_refinement=settings, **kwargs)
        from brats.innovations.detail_fusion.model import DetailFusionVMUNet
        return DetailFusionVMUNet(detail_fusion=settings, **kwargs)
    return VMUNet(**kwargs)


def build_criterion(config):
    extension = innovation_settings(config)
    # Every run optimizes the original CE + regional Dice; only the detail-fusion
    # boundary option adds an auxiliary term on its own training output.
    if extension is not None and extension[0] == "detail_fusion" and extension[1]["boundary_weight"] > 0:
        from brats.innovations.detail_fusion.loss import BoundaryAuxLoss
        return BoundaryAuxLoss(extension[1]["boundary_weight"])
    from brats.losses import BraTSCEDiceLoss
    return BraTSCEDiceLoss()


def validate_config(config):
    innovation_settings(config)
    warm_start_settings(config)
    EarlyStopping(**config.get("early_stopping", {}))
    if config["input_channels"] != 4 or config["num_classes"] != 4:
        raise ValueError("This protocol requires four MRI inputs and four exclusive output classes")
    if config["input_size"] < 32 or config["input_size"] % 32:
        raise ValueError("input_size must be divisible by 32")
    for key in ("epochs", "batch_size", "inference_batch_size", "val_interval"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_workers"] < 0 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer or worker configuration")
    if len(config["depths"]) != 4 or len(config["depths_decoder"]) != 4:
        raise ValueError("VM-UNet requires four encoder/decoder stages")


def load_training_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    # Machine-specific launch settings are independent of the training protocol.
    config.pop("launch", None)
    extension = innovation_settings(config)
    if extension is not None:
        # Persist resolved defaults in config.json/checkpoints for exact resume.
        config[extension[0]] = extension[1]
    warm_start = warm_start_settings(config)
    if warm_start is not None:
        config["warm_start"] = warm_start
    return config


def resolve_pretrained_path(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('pretrained must be null or a nonempty checkpoint path')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else SOURCE / path).resolve()


def resolve_warm_start_path(settings):
    """Absolute warm-start checkpoint path; relative paths start at the source directory."""
    if settings is None:
        return None
    path = Path(settings["checkpoint"]).expanduser()
    return (path if path.is_absolute() else SOURCE / path).resolve()


def apply_warm_start(model, settings, manifest_sha256, new_prefixes=()):
    """Load an earlier run's complete model; only the declared new tensors may stay new.

    The source must use the same patient split. Unexpected or reshaped tensors,
    missing tensors outside new_prefixes, and source tensors inside new_prefixes
    are errors, so "new" means exactly "absent from the source".
    """
    path = Path(settings["checkpoint"])
    source = load_checkpoint(path)
    if not isinstance(source, dict) or not isinstance(source.get("model_state_dict"), dict):
        raise ValueError("Warm start requires a training checkpoint containing model_state_dict")
    if source.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Warm-start checkpoint uses a different patient split")
    present = sorted(key for key in source["model_state_dict"]
                     if new_prefixes and key.startswith(tuple(new_prefixes)))
    if present:
        raise ValueError("Warm-start checkpoint already contains tensors declared new: " + ", ".join(present[:5]))
    result = model.load_state_dict(source["model_state_dict"], strict=False)
    if result.unexpected_keys:
        raise ValueError("Warm-start checkpoint has tensors absent from this model: "
                         + ", ".join(sorted(result.unexpected_keys)[:5]))
    outside = sorted(key for key in result.missing_keys if not key.startswith(tuple(new_prefixes)))
    if outside:
        raise ValueError("Warm-start checkpoint lacks original model tensors: " + ", ".join(outside[:5]))
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    report = {"source_path": str(path), "source_sha256": checksum.hexdigest(),
              "source_epoch": source.get("epoch"), "source_best_score": source.get("best_score"),
              "loaded_tensors": len(source["model_state_dict"]),
              "new_tensors": sorted(result.missing_keys)}
    print(f"Warm start: {path} (epoch {report['source_epoch']}); loaded={report['loaded_tensors']}, "
          f"new={len(report['new_tensors'])}", flush=True)
    return report


def optimizer_parameter_groups(model, config):
    """None keeps one parameter group; warm-start screening may give extensions their own LR."""
    settings = config.get("warm_start")
    prefixes = new_parameter_prefixes(config)
    if settings is None or settings["new_parameter_lr"] is None or not prefixes:
        return None
    named = list(model.named_parameters())
    new = [parameter for name, parameter in named if name.startswith(prefixes)]
    if not new:
        raise ValueError("The configured extension added no parameters")
    original = [parameter for name, parameter in named if not name.startswith(prefixes)]
    return [{"params": original}, {"params": new, "lr": settings["new_parameter_lr"]}]


def capture_rng(generator):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "loader": generator.get_state()}


def restore_rng(state, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if isinstance(state["cuda"], list):  # Legacy single-process checkpoint
        torch.cuda.set_rng_state_all(state["cuda"])
    else:
        torch.cuda.set_rng_state(state["cuda"])
    generator.set_state(state["loader"])


def save_checkpoint(path, checkpoint):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run(args, rank=0, world_size=1, local_rank=0):
    distributed = world_size > 1
    primary = rank == 0
    if args.mode in ("evaluate", "predict") and args.checkpoint is None:
        raise ValueError("Evaluation/prediction requires --checkpoint")
    if args.mode != "train" and args.resume:
        raise ValueError("--resume is only supported for training")
    if args.mode in ("train", "smoke") and args.checkpoint:
        raise ValueError("Use --resume for training or config.pretrained for encoder initialization")
    manifest = load_manifest(args.manifest)
    if manifest["purpose"] != "training":
        raise ValueError("--manifest must be the labeled training split")
    prediction_manifest = load_manifest(args.prediction_manifest) if args.mode == "predict" else None
    if prediction_manifest is not None and prediction_manifest["purpose"] != "prediction":
        raise ValueError("--prediction-manifest must contain only the official prediction cohort")
    checkpoint = None
    if args.checkpoint or args.resume:
        checkpoint = load_checkpoint(args.checkpoint or args.resume)
        if args.resume and checkpoint.get("evaluation_only", False):
            raise ValueError("Exported inference checkpoints cannot resume training; use the original latest.pth")
        if checkpoint["manifest_sha256"] != manifest["sha256"]:
            raise ValueError("Checkpoint and manifest use different patient splits")
    requested_config = load_training_config(args.config)
    pretrained_path = resolve_pretrained_path(requested_config["pretrained"])
    if pretrained_path is not None:
        requested_config["pretrained"] = str(pretrained_path)
        if checkpoint is None and not pretrained_path.is_file():
            raise FileNotFoundError(f'Pretrained checkpoint not found: {pretrained_path}')
    warm_start_path = resolve_warm_start_path(requested_config.get("warm_start"))
    if warm_start_path is not None:
        requested_config["warm_start"]["checkpoint"] = str(warm_start_path)
        if checkpoint is None and not warm_start_path.is_file():
            raise FileNotFoundError(f'Warm-start checkpoint not found: {warm_start_path}')
    config = checkpoint["config"] if args.mode in ("evaluate", "predict") else requested_config
    validate_config(config)
    if args.resume and config != checkpoint["config"]:
        raise ValueError("Resume requires the original config, including epochs and seed")
    if args.resume and checkpoint.get("world_size", 1) != world_size:
        raise ValueError("Resume requires the original number of training processes/GPUs")
    if args.mode == "train":
        if args.work_dir.exists() and any(args.work_dir.iterdir()) and not args.resume:
            raise FileExistsError("Work directory is not empty; choose a new --work-dir or use --resume")
        if args.resume and args.resume.resolve().parent != args.work_dir.resolve():
            raise ValueError("Resume in the original --work-dir to preserve best.pth and logs")
        if args.resume and not (args.work_dir / "best.pth").is_file():
            raise FileNotFoundError("Resume requires the original best.pth alongside latest.pth")
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from brats.dataset import BraTSSlices
    from brats.engine import evaluate

    if distributed:
        dist.barrier()  # All ranks must finish work-directory checks before any writes.
    seed = config["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda", local_rank)
    model = build_model(config)
    pretrained_report = warm_start_report = None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif config.get("warm_start") is not None:
        warm_start_report = apply_warm_start(model, config["warm_start"], manifest["sha256"],
                                             new_parameter_prefixes(config))
    elif config["pretrained"]:
        pretrained_report = model.load_from()
    model.to(device)
    raw_model = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    context = input_context(config)  # training and every 3D inference must stack slices alike
    if args.mode in ("evaluate", "predict"):
        split = "official_val" if args.mode == "predict" else args.split
        summary = evaluate(model, prediction_manifest if args.mode == "predict" else manifest,
                           args.cache_dir, split, device, config,
                           args.work_dir / ("evaluation_" + split),
                           save_predictions=args.save_predictions or args.mode == "predict",
                           context_slices=context)
        summary["checkpoint"] = str(args.checkpoint.resolve())
        summary["checkpoint_epoch"] = checkpoint["epoch"]
        summary["training_manifest_sha256"] = manifest["sha256"]
        summary["git"] = git_state(SOURCE)
        write_json(args.work_dir / ("evaluation_" + split) / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return
    dataset = BraTSSlices(manifest, args.cache_dir, "train", config["input_size"], context_slices=context)
    generator = torch.Generator().manual_seed(seed)
    # Padding at most world_size-1 slices keeps every rank's step count equal.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 seed=config["seed"], shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=sampler is None,
                        sampler=sampler,
                        num_workers=config["num_workers"], pin_memory=True,
                        multiprocessing_context="spawn" if distributed and config["num_workers"] > 0 else None,
                        worker_init_fn=seed_worker, generator=generator)
    criterion = build_criterion(config)
    component_names = getattr(criterion, "component_names", ("ce", "dice"))
    if args.mode == "smoke":
        batch = next(iter(loader))
        model.train()
        outputs = model(batch["image"].to(device))
        loss = criterion(outputs, batch["label"].to(device))
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite smoke gradients")
        print(f"[rank {rank}] Smoke forward/backward:", tuple(logits.shape), "loss:", loss.item(), flush=True)
        model.zero_grad(set_to_none=True)
        if primary:
            evaluate(raw_model, manifest, args.cache_dir, "val", device, config,
                     args.work_dir / "smoke", limit=1, context_slices=context)
        if distributed:
            dist.barrier()
        return

    parameter_groups = optimizer_parameter_groups(raw_model, config)
    optimizer = torch.optim.AdamW(model.parameters() if parameter_groups is None else parameter_groups,
                                  lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = GradScaler(enabled=config["amp"])
    start_epoch, best_score, best_epoch = 1, -1.0, 0
    early_stopping = EarlyStopping(**config.get("early_stopping", {}))
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch, best_score, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["best_epoch"]
        rng_state = (checkpoint["rng_state_by_rank"][rank] if "rng_state_by_rank" in checkpoint
                     else checkpoint["rng_state"])
        restore_rng(rng_state, generator)
        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])
        if early_stopping.stopped:
            if primary:
                print("This run has already early-stopped; use best.pth for evaluation.", flush=True)
            return
    if primary:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    if primary and not args.resume:
        write_json(args.work_dir / "config.json", config)
        write_json(args.work_dir / "split.json", manifest)
        # Research archives and generated artifacts are not experiment source.
        excluded = {"__pycache__", "tests", "archive", "results"}
        code_files = sorted(p for p in SOURCE.rglob("*.py")
                            if not any(part in excluded or part.startswith(".test_tmp")
                                       for part in p.relative_to(SOURCE).parts))
        write_json(args.work_dir / "run.json", {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "numpy": np.__version__,
            "manifest_sha256": manifest["sha256"], "parameters": sum(p.numel() for p in model.parameters()),
            "pretrained_report": pretrained_report, "warm_start_report": warm_start_report,
            "world_size": world_size, "batch_size_per_gpu": config["batch_size"],
            "global_batch_size": config["batch_size"] * world_size,
            "sampler_padding_per_epoch": len(sampler) * world_size - len(dataset) if sampler else 0,
            "code_sha256": {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
            "cache_dir": str(args.cache_dir.resolve()), "git": git_state(SOURCE)})
    if distributed:
        dist.barrier()
    if primary:
        print(f"Training with {world_size} GPU(s); batch/GPU={config['batch_size']}; "
              f"global batch={config['batch_size'] * world_size}; lr={config['learning_rate']}", flush=True)
    for epoch in range(start_epoch, config["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss, n_samples = 0.0, 0
        component_totals = torch.zeros(len(component_names), dtype=torch.float64, device=device)
        start = time.perf_counter()
        for step, batch in enumerate(loader, 1):
            images, labels = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config["amp"]):
                loss, components = criterion(model(images), labels, return_components=True)
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise ValueError(f"Nonfinite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(images)
            # Detached GPU accumulation adds no per-step component .item() sync.
            component_totals += torch.stack([components[name] for name in component_names]) * len(images)
            n_samples += len(images)
            if primary and (step == 1 or step % 50 == 0):
                print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.5f}", flush=True)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        totals = torch.cat((torch.tensor([total_loss, n_samples], dtype=torch.float64, device=device),
                            component_totals))
        elapsed = torch.tensor(seconds, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(totals)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        record = {"epoch": epoch, "train_loss": (totals[0] / totals[1]).item(),
                  **{f"train_{name}_loss": (totals[index + 2] / totals[1]).item()
                     for index, name in enumerate(component_names)},
                  "learning_rate": optimizer.param_groups[0]["lr"], "train_seconds": elapsed.item()}
        improved = False
        if epoch == 1 or epoch % config["val_interval"] == 0 or epoch == config["epochs"]:
            if primary:
                # Bypass the DDP wrapper so other ranks can wait without forward collectives.
                summary = evaluate(raw_model, manifest, args.cache_dir, "val", device, config,
                                   args.work_dir / "validation" / f"epoch_{epoch:03d}",
                                   context_slices=context)
                record["val_mean_dice"] = summary["mean_dice"]
                record["val_dice"] = {r: summary["regions"][r]["dice"]["mean"] for r in ("WT", "TC", "ET")}
                early_stopping.update(summary["mean_dice"])
                # Preserve true maximum-Dice selection independently of min_delta.
                if summary["mean_dice"] > best_score:
                    best_score, best_epoch, improved = summary["mean_dice"], epoch, True
            if distributed:
                status = [best_score, best_epoch, improved, early_stopping.state_dict()]
                dist.broadcast_object_list(status, src=0)
                best_score, best_epoch, improved, stopping_state = status
                early_stopping.load_state_dict(stopping_state)
        record["early_stopping"] = early_stopping.state_dict()
        record["best_epoch"] = best_epoch
        record["best_val_mean_dice"] = best_score
        scheduler.step()
        local_rng = capture_rng(generator)
        rng_states = [None] * world_size
        if distributed:
            dist.all_gather_object(rng_states, local_rng)
        else:
            rng_states = [local_rng]
        if primary:
            state = {"epoch": epoch, "config": config, "manifest_sha256": manifest["sha256"],
                     "model_state_dict": raw_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                     "best_score": best_score, "best_epoch": best_epoch,
                     "world_size": world_size, "rng_state_by_rank": rng_states,
                     "early_stopping_state": early_stopping.state_dict()}
            if improved:
                save_checkpoint(args.work_dir / "best.pth", state)
            save_checkpoint(args.work_dir / "latest.pth", state)
            with (args.work_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({**record, "best_val_mean_dice": best_score, "best_epoch": best_epoch}), flush=True)
        if distributed:
            dist.barrier()  # Complete checkpoint writes before another epoch or teardown.
        if early_stopping.stopped:
            if primary:
                print(f"Early stopping at epoch {epoch}: {early_stopping.bad_validations} validation checks "
                      f"without an increase > {early_stopping.min_delta}.", flush=True)
            break
    # Keep the held-out test untouched until an explicit --mode evaluate request.
    if primary:
        print(f"Training finished. Best validation Mean Dice={best_score:.5f} at epoch {best_epoch}.", flush=True)


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if world_size > 1 and args.mode not in ("train", "smoke"):
        raise ValueError("Use one GPU (python, not multi-process torchrun) for evaluation/prediction")
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError("VM-UNet selective scan requires a configured CUDA training environment")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        # Rank zero performs full 3D validation while the remaining ranks wait.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    try:
        run(args, rank, world_size, local_rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
