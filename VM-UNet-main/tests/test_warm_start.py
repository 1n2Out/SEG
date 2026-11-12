"""Warm-start screening: settings, safe loading, optimizer groups and launcher checks."""
import json

import pytest

import run_brats
from train_brats import (SOURCE, load_training_config, optimizer_parameter_groups, validate_config,
                         warm_start_settings)

SCREENING = ("baseline", "detail_gated", "detail_concat", "shallow_gated")


def screening_config(name):
    return json.loads((SOURCE / "configs/screening" / (name + ".json")).read_text(encoding="utf-8"))


@pytest.mark.parametrize("settings", [None, "best.pth", {}, {"checkpoint": ""}, {"checkpoint": 3},
    {"checkpoint": "best.pth", "lr": 1e-4}, {"checkpoint": "best.pth", "new_parameter_lr": 0},
    {"checkpoint": "best.pth", "new_parameter_lr": True},
    {"checkpoint": "best.pth", "new_parameter_lr": float("nan")}])
def test_invalid_warm_start_rejected(settings):
    with pytest.raises(ValueError, match="warm_start"):
        warm_start_settings({"pretrained": None, "warm_start": settings})


def test_warm_start_replaces_pretrained_initialization():
    with pytest.raises(ValueError, match="pretrained"):
        warm_start_settings({"pretrained": "weights.pth", "warm_start": {"checkpoint": "best.pth"}})
    assert warm_start_settings({"pretrained": "weights.pth"}) is None
    assert warm_start_settings({"pretrained": None, "warm_start": {"checkpoint": "best.pth"}}) == {
        "checkpoint": "best.pth", "new_parameter_lr": None}


def test_screening_configs_share_one_protocol_and_differ_only_by_extension():
    formal = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    gpu_count = len(formal["launch"].pop("gpus"))
    protocols = []
    for name in SCREENING:
        validate_config(load_training_config(SOURCE / "configs/screening" / (name + ".json")))
        config = screening_config(name)
        assert config["launch"].pop("work_dir") == "results/screening_" + name
        # Runs may sit on different GPUs in parallel; each keeps the formal GPU count (batch 16).
        assert len(config["launch"].pop("gpus")) == gpu_count
        config.pop("detail_fusion", None)
        config.pop("shallow_refinement", None)
        protocols.append(config)
    assert all(protocol == protocols[0] for protocol in protocols)
    screening = dict(protocols[0])
    assert screening.pop("warm_start") == {"checkpoint": "results/brats2020_1102_1734/best.pth",
                                           "new_parameter_lr": 0.0001}
    assert (screening.pop("epochs"), screening.pop("learning_rate"), screening.pop("pretrained")) == (10, 1e-05, None)
    del formal["launch"]["work_dir"]
    for key in ("epochs", "learning_rate", "pretrained"):
        del formal[key]
    assert screening == formal  # everything else follows the EXP-001 protocol
    assert set(screening_config("baseline")) & {"detail_fusion", "shallow_refinement"} == set()
    assert screening_config("detail_gated")["detail_fusion"]["mode"] == "gated"
    assert screening_config("detail_concat")["detail_fusion"]["mode"] == "concat"
    assert screening_config("shallow_gated")["shallow_refinement"] == {"mode": "gated", "hidden_channels": 24}


torch = pytest.importorskip("torch")
from train_brats import apply_warm_start, save_checkpoint


class Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vmunet = torch.nn.Linear(3, 2)


class Extended(Original):
    def __init__(self):
        super().__init__()
        self.detail_fusion = torch.nn.Linear(2, 2)


def write_source(path, state, manifest="split-a"):
    save_checkpoint(path, {"model_state_dict": state, "manifest_sha256": manifest,
                           "epoch": 55, "best_score": 0.8178})
    return {"checkpoint": str(path), "new_parameter_lr": 1e-4}


def test_loads_original_tensors_and_keeps_new_extension_initialization(tmp_path):
    torch.manual_seed(0)
    source = Original()
    settings = write_source(tmp_path / "best.pth", source.state_dict())
    model = Extended()
    initial = model.detail_fusion.weight.detach().clone()
    report = apply_warm_start(model, settings, "split-a", ("detail_fusion.",))
    assert torch.equal(model.vmunet.weight, source.vmunet.weight)
    assert torch.equal(model.detail_fusion.weight, initial)
    assert report["new_tensors"] == ["detail_fusion.bias", "detail_fusion.weight"]
    assert (report["source_epoch"], report["source_best_score"], report["loaded_tensors"]) == (55, 0.8178, 2)
    assert len(report["source_sha256"]) == 64
    control = Original()  # the baseline control loads every tensor and adds none
    assert apply_warm_start(control, settings, "split-a")["new_tensors"] == []
    assert torch.equal(control.vmunet.bias, source.vmunet.bias)


def test_rejects_other_splits_and_incompatible_tensors(tmp_path):
    original = write_source(tmp_path / "original.pth", Original().state_dict())
    with pytest.raises(ValueError, match="different patient split"):
        apply_warm_start(Extended(), original, "split-b", ("detail_fusion.",))
    with pytest.raises(ValueError, match="lacks original"):
        apply_warm_start(Extended(), original, "split-a")  # extension tensors not declared as new
    extended = write_source(tmp_path / "extended.pth", Extended().state_dict())
    with pytest.raises(ValueError, match="absent from this model"):
        apply_warm_start(Original(), extended, "split-a")
    reshaped = write_source(tmp_path / "reshaped.pth",
                            {"vmunet.weight": torch.zeros(4, 3), "vmunet.bias": torch.zeros(4)})
    with pytest.raises(RuntimeError, match="size mismatch"):
        apply_warm_start(Original(), reshaped, "split-a")
    bare = tmp_path / "weights_only.pth"
    torch.save(Original().state_dict(), bare)
    with pytest.raises(ValueError, match="model_state_dict"):
        apply_warm_start(Original(), {"checkpoint": str(bare)}, "split-a")


def test_new_extension_parameters_get_their_own_annealed_learning_rate():
    model = Extended()
    config = {"pretrained": None, "detail_fusion": {"mode": "gated"},
              "warm_start": {"checkpoint": "best.pth", "new_parameter_lr": 1e-4}}
    groups = optimizer_parameter_groups(model, config)
    assert [len(group["params"]) for group in groups] == [2, 2]
    assert groups[0]["params"][0] is model.vmunet.weight
    assert groups[1]["params"][0] is model.detail_fusion.weight
    optimizer = torch.optim.AdamW(groups, lr=1e-5, weight_decay=0.01)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[1]["lr"] == pytest.approx(10 * optimizer.param_groups[0]["lr"])
    # One group otherwise: disabled extension, baseline control, formal training, shared rate.
    for other in ({**config, "detail_fusion": {"mode": "off"}},
                  {key: value for key, value in config.items() if key != "detail_fusion"},
                  {"pretrained": None, "detail_fusion": {"mode": "gated"}},
                  {**config, "warm_start": {"checkpoint": "best.pth", "new_parameter_lr": None}}):
        assert optimizer_parameter_groups(model, other) is None


@pytest.fixture
def screening_path(tmp_path):
    config = screening_config("detail_gated")
    config["launch"].update({"gpus": [0], "work_dir": str(tmp_path / "screen"),
                             "cache_dir": str(tmp_path / "cache"), "manifest": str(tmp_path / "split.json"),
                             "data_root": str(tmp_path / "raw"), "log_dir": str(tmp_path / "logs")})
    config["warm_start"]["checkpoint"] = str(tmp_path / "baseline_run" / "best.pth")
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_launcher_requires_warm_start_file_only_for_new_runs(screening_path, tmp_path):
    for action in ("train", "smoke"):
        plan = run_brats.make_plan(action, screening_path)
        plan["manifest"].touch()
        plan["cache_dir"].mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Warm-start checkpoint not found"):
            run_brats.check_inputs(plan)
        assert not plan["work_dir"].exists() and not plan["log_file"].exists()
    (tmp_path / "baseline_run").mkdir()
    (tmp_path / "baseline_run" / "best.pth").touch()
    run_brats.check_inputs(run_brats.make_plan("train", screening_path))
    (tmp_path / "baseline_run" / "best.pth").unlink()
    training = tmp_path / "existing_run"
    training.mkdir()
    for name in ("best.pth", "latest.pth"):
        (training / name).touch()
    for action in ("resume", "test"):  # these use the run's own checkpoints
        run_brats.check_inputs(run_brats.make_plan(action, screening_path, training))


def test_relative_warm_start_path_starts_at_source_directory():
    plan = run_brats.make_plan("train", SOURCE / "configs/screening/detail_gated.json")
    assert plan["warm_start_path"] == (SOURCE / "results/brats2020_1102_1734/best.pth").resolve()
    formal = run_brats.make_plan("train", SOURCE / "configs/detail_fusion/gated.json")
    assert formal["warm_start_path"] is None and formal["pretrained_path"] is not None
