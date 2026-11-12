"""git_state reads commit, branch and uncommitted changes from a temporary repository."""
import shutil
import subprocess

import pytest

from brats.provenance import code_changes_since, git_state

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_git_state_reports_commit_branch_and_changes(tmp_path, monkeypatch):
    # The pytest base directory may sit inside the project repository.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert git_state(tmp_path / "missing") is None
    repo = tmp_path / "repo"
    repo.mkdir()
    assert git_state(repo) is None  # Not a repository yet.
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    (repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    (repo / "model.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    (repo / "results").mkdir()
    (repo / "results" / "log.txt").write_text("ignored", encoding="utf-8")
    state = git_state(repo)
    assert len(state["commit"]) == 40 and state["branch"] == "main"
    assert state == {**state, "dirty": False, "changes": []}
    (repo / "new_module.py").write_text("y = 2\n", encoding="utf-8")
    assert git_state(repo)["dirty"] and "new_module.py" in git_state(repo)["changes"][0]
    (repo / "new_module.py").unlink()
    (repo / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert git_state(repo)["changes"] == [" M model.py"]


def test_code_changes_ignore_docs_tests_analysis_and_other_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, source = tmp_path / "repo", tmp_path / "repo" / "src"
    for name in ("brats/model.py", "tests/test_model.py", "analysis/stats.py", "README.md", "docs/guide.md"):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("v1\n", encoding="utf-8")
    (repo / "STATE.md").write_text("v1\n", encoding="utf-8")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "run")
    commit = git_state(repo)["commit"]
    assert code_changes_since(source, commit) == []
    assert code_changes_since(source, "f" * 40) is None  # Unknown or unfetched commit.
    # Reports, docs and tests change after the run; a new HEAD alone is not a code change.
    for name in ("tests/test_model.py", "analysis/stats.py", "README.md", "docs/guide.md"):
        (source / name).write_text("v2\n", encoding="utf-8")
    (repo / "STATE.md").write_text("v2\n", encoding="utf-8")
    git(repo, "commit", "-qam", "reports")
    (source / "analysis" / "new.py").write_text("x\n", encoding="utf-8")
    assert code_changes_since(source, commit) == []
    (source / "brats" / "model.py").write_text("v2\n", encoding="utf-8")
    (source / "brats" / "extra.py").write_text("new\n", encoding="utf-8")
    assert code_changes_since(source, commit) == ["src/brats/extra.py", "src/brats/model.py"]
