"""Launcher checks without CUDA, training, or a detached training process."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import run_brats
from train_brats import SOURCE, load_training_config


@pytest.fixture
def config_path(tmp_path):
    config = json.loads((SOURCE / "configs/brats2020.json").read_text(encoding="utf-8"))
    config["pretrained"] = None  # Launcher tests do not require external model weights.
    # Exercise two-GPU launch behavior independently of this machine's defaults.
    config["launch"].update({"gpus": [0, 1],
                             "work_dir": str(tmp_path / "experiment with spaces"),
                             "cache_dir": str(tmp_path / "cache"),
                             "manifest": str(tmp_path / "split.json"),
                             "data_root": str(tmp_path / "raw"), "log_dir": str(tmp_path / "logs")})
    path = tmp_path / "config with spaces.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def change_setting(path, key, value):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["launch"][key] = value
    path.write_text(json.dumps(config), encoding="utf-8")


def test_single_and_multi_gpu_commands(config_path):
    plan = run_brats.make_plan("train", config_path)
    assert plan["command"][:4] == [sys.executable, "-u", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in plan["command"]
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    change_setting(config_path, "gpus", [3])
    plan = run_brats.make_plan("train", config_path)
    assert "torch.distributed.run" not in plan["command"]
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert str(config_path) in plan["command"]  # A space remains part of one argv item.


def test_resume_and_single_gpu_test_use_selected_experiment(config_path):
    change_setting(config_path, "gpus", [2, 4])
    train = run_brats.make_plan("train", config_path)
    resume = run_brats.make_plan("resume", config_path, train["work_dir"])
    test = run_brats.make_plan("test", config_path, train["work_dir"])
    assert resume["command"][-2:] == ["--resume", str(resume["work_dir"] / "latest.pth")]
    assert resume["environment"]["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert "torch.distributed.run" not in test["command"]
    assert test["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert str(train["work_dir"] / "best.pth") in test["command"]
    assert test["work_dir"].parent == train["work_dir"]
    assert "evaluate" in test["command"] and "--save-predictions" in test["command"]
    assert resume["log_file"] == train["log_file"]
    assert test["log_file"] != resume["log_file"]


def test_smoke_output_cannot_make_training_directory_nonempty(config_path):
    plan = run_brats.make_plan("smoke", config_path)
    output = Path(plan["command"][plan["command"].index("--work-dir") + 1])
    assert plan["base_dir"] not in output.parents and output != plan["base_dir"]
    assert output.name.startswith(plan["base_dir"].name + "_smoke_")


def test_preparation_preserves_fixed_split_seed_independent_of_training_seed(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = 123
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = run_brats.make_plan("prepare", config_path)
    assert "torch.distributed.run" not in plan["command"]
    assert str(SOURCE / "prepare_brats.py") in plan["command"]
    assert "--seed" not in plan["command"]  # prepare_brats retains fixed split seed=42.
    assert plan["command"][-1] == str(plan["cache_dir"])


def test_launch_settings_do_not_change_checkpoint_training_protocol(config_path):
    original = load_training_config(config_path)
    change_setting(config_path, "gpus", [4, 5])
    change_setting(config_path, "log_dir", "another_logs")
    assert load_training_config(config_path) == original
    assert "launch" not in original


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1], [True], "0,1", ["0"]])
def test_reject_invalid_gpu_selection(config_path, gpus):
    change_setting(config_path, "gpus", gpus)
    with pytest.raises(ValueError, match="launch.gpus"):
        run_brats.make_plan("train", config_path)


def test_logs_cannot_populate_new_experiment_directory(config_path):
    plan = run_brats.make_plan("train", config_path)
    change_setting(config_path, "log_dir", str(plan["base_dir"] / "logs"))
    with pytest.raises(ValueError, match="outside"):
        run_brats.make_plan("train", config_path)


def test_preflight_rejects_missing_data_or_checkpoints_and_existing_run(config_path):
    plan = run_brats.make_plan("train", config_path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        run_brats.check_inputs(plan)
    plan["manifest"].touch()
    with pytest.raises(FileNotFoundError, match="cache"):
        run_brats.check_inputs(plan)
    plan["cache_dir"].mkdir()
    run_brats.check_inputs(plan)
    plan["work_dir"].mkdir()
    (plan["work_dir"] / "history.jsonl").touch()
    with pytest.raises(FileExistsError, match="already exists"):
        run_brats.check_inputs(plan)
    resume = run_brats.make_plan("resume", config_path, plan["work_dir"])
    with pytest.raises(FileNotFoundError, match="latest.pth"):
        run_brats.check_inputs(resume)
    for name in ("latest.pth", "best.pth"):
        (plan["work_dir"] / name).touch()
    run_brats.check_inputs(resume)


def test_background_detaches_redirects_and_appends_without_starting_job(config_path, monkeypatch):
    train = run_brats.make_plan("train", config_path)
    plan = run_brats.make_plan("resume", config_path, train["work_dir"])
    plan["log_file"].parent.mkdir()
    plan["log_file"].write_bytes(b"previous log\n")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"new log\n")
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(run_brats.subprocess, "Popen", fake_popen)
    assert run_brats.start_background(plan) == 12345
    assert plan["log_file"].read_bytes() == b"previous log\nnew log\n"
    assert plan["log_file"].with_suffix(".pid").read_text().strip() == "12345"
    command, kwargs = calls[0]
    assert command == plan["command"]
    assert kwargs["start_new_session"] and kwargs["close_fds"]
    assert kwargs["stdin"] == run_brats.subprocess.DEVNULL
    assert kwargs["stderr"] == run_brats.subprocess.STDOUT
    assert kwargs["cwd"] == str(SOURCE)
    assert not plan["work_dir"].exists()


def test_dry_run_never_spawns_or_creates_logs(config_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_brats.py", "--config", str(config_path), "--background", "--dry-run"])
    monkeypatch.setattr(run_brats, "start_background", lambda _: pytest.fail("Must not spawn"))
    run_brats.main()
    assert "torch.distributed.run" in capsys.readouterr().out
    plan = run_brats.make_plan("train", config_path)
    assert not plan["log_file"].parent.exists()


def test_foreground_exec_preserves_signal_handling(config_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_brats.py", "--config", str(config_path)])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(run_brats, "check_inputs", lambda _: None)
    monkeypatch.setattr(run_brats.os, "chdir", lambda path: calls.append(path))
    monkeypatch.setattr(run_brats.os, "execvpe", lambda *args: calls.append(args))
    run_brats.main()
    assert calls[0] == SOURCE
    assert calls[1][0] == sys.executable
    assert calls[1][2]["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize("action", ["train", "smoke", "test"])
@pytest.mark.parametrize("collision", ["directory", "log", "pid"])
def test_new_tasks_do_not_reuse_outputs_or_logs(config_path, monkeypatch, action, collision):
    monkeypatch.setattr(run_brats, "timestamp_label", lambda: "1102_1700")
    first = run_brats.make_plan(action, config_path)
    if collision == "directory":
        first["work_dir"].mkdir(parents=True)
    else:
        first["log_file"].parent.mkdir(parents=True)
        path = first["log_file"] if collision == "log" else first["log_file"].with_suffix(".pid")
        path.write_text("original")
    second = run_brats.make_plan(action, config_path)
    assert first["work_dir"] != second["work_dir"]
    assert first["log_file"] != second["log_file"]
    assert second["work_dir"].name.endswith("_01")
    assert not second["work_dir"].exists()
    if collision != "directory":
        assert path.read_text() == "original"


def test_reservation_rejects_simultaneous_plans_without_reusing_directory(config_path, monkeypatch):
    monkeypatch.setattr(run_brats, "timestamp_label", lambda: "1102_1700")
    first = run_brats.make_plan("train", config_path)
    concurrent = run_brats.make_plan("train", config_path)
    run_brats.reserve_output(first)
    with pytest.raises(FileExistsError):
        run_brats.reserve_output(concurrent)
    assert list(first["work_dir"].iterdir()) == []


def test_multiple_runs_require_selection_and_explicit_legacy_resume_still_works(config_path):
    train = run_brats.make_plan("train", config_path)
    train["base_dir"].mkdir(parents=True)
    train["work_dir"].mkdir()
    for action in ("resume", "test"):
        with pytest.raises(ValueError, match="--run-dir"):
            run_brats.make_plan(action, config_path)
    resume = run_brats.make_plan("resume", config_path, train["base_dir"])
    assert resume["work_dir"] == train["base_dir"]
    assert resume["command"][-1] == str(train["base_dir"] / "latest.pth")


def test_single_dated_run_is_selected_but_smoke_and_test_are_not_candidates(config_path):
    train = run_brats.make_plan("train", config_path)
    train["work_dir"].mkdir(parents=True)
    smoke = run_brats.make_plan("smoke", config_path)
    smoke["work_dir"].mkdir()
    test = run_brats.make_plan("test", config_path)
    test["work_dir"].mkdir()
    assert run_brats.make_plan("resume", config_path)["work_dir"] == train["work_dir"]
    assert test["run_dir"] == train["work_dir"]


def test_test_checks_checkpoint_in_training_run_not_output_directory(config_path):
    train = run_brats.make_plan("train", config_path)
    train["work_dir"].mkdir(parents=True)
    test = run_brats.make_plan("test", config_path, train["work_dir"])
    test["manifest"].touch()
    test["cache_dir"].mkdir()
    with pytest.raises(FileNotFoundError, match="best.pth"):
        run_brats.check_inputs(test)
    (train["work_dir"] / "best.pth").touch()
    run_brats.check_inputs(test)
    assert not test["work_dir"].exists()


def test_new_background_log_cannot_overwrite_or_append_existing_log(config_path, monkeypatch):
    plan = run_brats.make_plan("train", config_path)
    plan["log_file"].parent.mkdir()
    plan["log_file"].write_text("original log")
    monkeypatch.setattr(run_brats.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not spawn"))
    with pytest.raises(FileExistsError):
        run_brats.start_background(plan)
    assert plan["log_file"].read_text() == "original log"


def test_two_background_training_launches_are_isolated(config_path, monkeypatch):
    monkeypatch.setattr(run_brats, "timestamp_label", lambda: "1102_1700")
    initial = run_brats.make_plan("train", config_path)
    initial["manifest"].touch()
    initial["cache_dir"].mkdir()
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(b"training startup\n")
        return SimpleNamespace(pid=1000 + len(calls))

    monkeypatch.setattr(run_brats.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", ["run_brats.py", "--config", str(config_path), "--background"])
    monkeypatch.setattr(sys, "platform", "linux")
    run_brats.main()
    run_brats.main()
    outputs = [Path(cmd[cmd.index("--work-dir") + 1]) for cmd in calls]
    assert len(set(outputs)) == 2 and all(path.is_dir() for path in outputs)
    for path in outputs:
        log = initial["log_file"].parent / (path.name + ".log")
        assert log.read_bytes() == b"training startup\n"
        assert log.with_suffix(".pid").is_file()


def test_run_dir_is_only_used_to_select_existing_training(config_path):
    with pytest.raises(ValueError, match="only supported"):
        run_brats.make_plan("train", config_path, "some_run")


def test_timestamp_uses_shanghai_offset(monkeypatch):
    from datetime import datetime, timedelta

    class FixedDateTime:
        @staticmethod
        def now(zone):
            assert zone.utcoffset(None) == timedelta(hours=8)
            return datetime(2026, 11, 2, 17, 0, 0, 123456)

    monkeypatch.setattr(run_brats, "datetime", FixedDateTime)
    assert run_brats.timestamp_label() == "1102_1700"


@pytest.mark.parametrize('action', ['resume', 'test'])
@pytest.mark.parametrize('suffix', ['20261102_170158_608727', '20261102_170158_608727_01'])
def test_old_timestamp_directories_remain_selectable(config_path, action, suffix):
    base = run_brats.make_plan('train', config_path)['base_dir']
    old = base.with_name(base.name + '_' + suffix)
    old.mkdir(parents=True)
    plan = run_brats.make_plan(action, config_path)
    assert plan['run_dir'] == old
    name = 'latest.pth' if action == 'resume' else 'best.pth'
    assert str(old / name) in plan['command']


@pytest.mark.parametrize('action', ['resume', 'test'])
def test_old_and_short_names_cannot_silently_select_wrong_checkpoint(config_path, action):
    base = run_brats.make_plan('train', config_path)['base_dir']
    short = base.with_name(base.name + '_1102_1700_01')
    old = base.with_name(base.name + '_20261102_170158_608727')
    short.mkdir(parents=True)
    old.mkdir()
    with pytest.raises(ValueError, match='--run-dir'):
        run_brats.make_plan(action, config_path)
    for directory in (short, old):
        assert run_brats.make_plan(action, config_path, directory)['run_dir'] == directory


def test_missing_pretraining_fails_before_creating_a_new_run(config_path, tmp_path):
    config = json.loads(config_path.read_text())
    config['pretrained'] = str(tmp_path / 'absent_weights.pth')
    config_path.write_text(json.dumps(config))
    for action in ('train', 'smoke'):
        plan = run_brats.make_plan(action, config_path)
        plan['manifest'].touch()
        plan['cache_dir'].mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match='Pretrained checkpoint not found'):
            run_brats.check_inputs(plan)
        assert not plan['work_dir'].exists()
        assert not plan['log_file'].exists()


def test_resume_and_evaluation_do_not_require_original_pretraining_file(config_path, tmp_path):
    config = json.loads(config_path.read_text())
    config['pretrained'] = str(tmp_path / 'absent_weights.pth')
    config_path.write_text(json.dumps(config))
    training = tmp_path / 'existing_run'
    training.mkdir()
    for name in ('best.pth', 'latest.pth'):
        (training / name).touch()
    for action in ('resume', 'test'):
        plan = run_brats.make_plan(action, config_path, training)
        plan['manifest'].touch()
        plan['cache_dir'].mkdir(exist_ok=True)
        run_brats.check_inputs(plan)
