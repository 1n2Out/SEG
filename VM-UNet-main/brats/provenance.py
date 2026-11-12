"""Git provenance of the source tree, recorded with every run and evaluation."""
import subprocess


def _git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def git_state(path):
    """Commit, branch and uncommitted changes of the repository containing path, or None."""
    try:
        commit = _git(path, "rev-parse", "HEAD").strip()
        branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip()
        # Untracked, non-ignored files count as changes: they may be imported code.
        changes = _git(path, "status", "--porcelain", "--untracked-files=all").splitlines()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return None
    return {"commit": commit, "branch": None if branch == "HEAD" else branch,
            "dirty": bool(changes), "changes": changes[:50]}


# Paths under the source directory which cannot change a run's results.
NON_CODE = (":(exclude)tests", ":(exclude)analysis", ":(exclude)*.md")


def code_changes_since(path, commit):
    """Code-relevant files changed between commit and the working tree, or None if unknown.

    Documentation and reports are committed constantly, so commit equality is too
    strict; only source files under path (tracked changes and new untracked files)
    count. None means git, the repository or the commit is unavailable.
    """
    try:
        _git(path, "cat-file", "-e", f"{commit}^{{commit}}")
        changed = _git(path, "diff", "--name-only", commit, "--", ".", *NON_CODE).splitlines()
        added = _git(path, "ls-files", "--others", "--exclude-standard", "--full-name",
                     "--", ".", *NON_CODE).splitlines()
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return None
    return sorted(set(changed) | set(added))
