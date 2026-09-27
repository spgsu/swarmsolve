"""Git plumbing for isolating a solve run on its own worktree/branch, and
auto-merging it back once a run is fully verified.

Stdlib-only (``subprocess``) -- no GitPython dependency. Every function shells
out to ``git`` directly; if git isn't installed, callers degrade gracefully
(see ``pipeline.py``) rather than failing the run -- isolation/verification/
auto-merge are mitigations, not a guarantee.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def git_available() -> bool:
    return shutil.which("git") is not None


def _run(args: list[str], cwd: Path, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )


def is_git_repo(path: Path) -> bool:
    """Return whether ``path`` itself is a repository/worktree root.

    ``git rev-parse --is-inside-work-tree`` also returns true for arbitrary
    child directories. That is useful interactively but wrong for callers
    deciding whether a plain project directory needs its own local repository.
    """
    if not git_available():
        return False
    root = repo_root(path)
    return root is not None and root.resolve() == path.resolve()


def repo_root(path: Path) -> Path | None:
    if not git_available():
        return None
    try:
        r = _run(["rev-parse", "--show-toplevel"], cwd=path)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def current_sha(path: Path) -> str | None:
    try:
        r = _run(["rev-parse", "HEAD"], cwd=path)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def has_uncommitted_changes(path: Path) -> bool:
    try:
        r = _run(["status", "--porcelain"], cwd=path)
    except (OSError, subprocess.TimeoutExpired):
        return True  # can't tell -- treat as dirty, the safe direction for a merge guard
    return bool(r.stdout.strip())


@dataclass
class WorktreeInfo:
    repo_root: Path
    workspace: Path
    branch: str
    base_sha: str


def create_run_worktree(repo: Path, workspace: Path, run_id: str, base_ref: str = "HEAD") -> WorktreeInfo:
    """Create a new branch ``solve/<run_id>`` and a worktree for it at
    ``workspace`` (must not already exist). Raises RuntimeError on failure --
    callers fall back to writing directly instead of half-isolating a run."""
    branch = f"solve/{run_id}"
    base_sha = current_sha(repo)
    if base_sha is None:
        raise RuntimeError(f"could not resolve {base_ref} in {repo}")
    try:
        r = _run(["worktree", "add", "-b", branch, str(workspace), base_ref], cwd=repo, timeout=60.0)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"git worktree add failed: {e}") from e
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {r.stderr.strip()}")
    return WorktreeInfo(repo_root=repo, workspace=workspace, branch=branch, base_sha=base_sha)


def init_fresh_repo(path: Path) -> str | None:
    """git init a brand-new repo at ``path`` with one empty commit, so
    verification has a base SHA to diff against even when the workspace isn't
    inside an existing project. Returns the base SHA, or None if git is
    unavailable/init failed (verification then runs without a diff base)."""
    if not git_available():
        return None
    path.mkdir(parents=True, exist_ok=True)
    if _run(["init", "-q"], cwd=path).returncode != 0:
        return None
    _run(["config", "user.email", "solve@localhost"], cwd=path)
    _run(["config", "user.name", "swarmsolve"], cwd=path)
    # Stage whatever is already in the directory so the base commit captures the
    # project's real files. Committing --allow-empty with NO `git add` would turn
    # a non-git project into a repo whose HEAD had an empty tree -- so every
    # worktree created from HEAD checked out empty. --allow-empty is kept so a
    # genuinely empty scratch dir still gets a base.
    _run(["add", "-A"], cwd=path)
    if _run(["commit", "--allow-empty", "-q", "-m", "solve: initial commit"], cwd=path).returncode != 0:
        return None
    return current_sha(path)


def diff_files(workspace: Path, base_sha: str) -> list[str]:
    """Files that differ from ``base_sha`` in ``workspace``'s working tree --
    tracked changes (including uncommitted) plus untracked new files, since
    agents rarely commit their own work mid-round."""
    names: set[str] = set()
    tracked = _run(["diff", "--name-only", base_sha], cwd=workspace)
    if tracked.returncode == 0:
        names.update(x for x in tracked.stdout.splitlines() if x)
    untracked = _run(["ls-files", "--others", "--exclude-standard"], cwd=workspace)
    if untracked.returncode == 0:
        names.update(x for x in untracked.stdout.splitlines() if x)
    return sorted(names)


@dataclass
class MergeResult:
    merged: bool
    branch: str
    commit_sha: str | None = None
    error: str | None = None


@dataclass
class CheckpointResult:
    committed: bool
    commit_sha: str | None = None
    error: str | None = None


def checkpoint_worktree(path: Path, message: str) -> CheckpointResult:
    """Commit pending run state without merging it anywhere.

    Failed solves remain recoverable and reviewable on their isolated branch;
    a successful solve can subsequently merge the same commit.
    """
    add = _run(["add", "-A"], cwd=path)
    if add.returncode != 0:
        return CheckpointResult(False, error=f"could not stage checkpoint: {add.stderr.strip()}")
    if not has_uncommitted_changes(path):
        return CheckpointResult(False, commit_sha=current_sha(path))
    commit = _run(["commit", "-q", "-m", message], cwd=path)
    if commit.returncode != 0:
        return CheckpointResult(False, error=f"could not commit checkpoint: {commit.stderr.strip()}")
    return CheckpointResult(True, commit_sha=current_sha(path))


def merge_worktree_branch(repo: Path, workspace: Path, branch: str) -> MergeResult:
    """Commit any pending changes in the worktree, then merge ``branch`` into
    whatever's currently checked out in ``repo``. Conservative by design:
    refuses if the target branch is dirty, prefers ``--ff-only``, and aborts
    immediately (never leaves a conflicted repo) if a plain merge isn't clean.
    """
    checkpoint = checkpoint_worktree(workspace, "solve: final state")
    if checkpoint.error:
        return MergeResult(merged=False, branch=branch, error=checkpoint.error)

    if has_uncommitted_changes(repo):
        return MergeResult(merged=False, branch=branch,
                           error="target branch has uncommitted changes; refusing to merge onto it")

    r = _run(["merge", "--ff-only", branch], cwd=repo, timeout=60.0)
    if r.returncode == 0:
        return MergeResult(merged=True, branch=branch, commit_sha=current_sha(repo))

    r = _run(["merge", "--no-edit", branch], cwd=repo, timeout=60.0)
    if r.returncode == 0:
        return MergeResult(merged=True, branch=branch, commit_sha=current_sha(repo))
    _run(["merge", "--abort"], cwd=repo)
    return MergeResult(merged=False, branch=branch,
                       error=f"merge conflict; branch left at {branch!r} for a manual merge")
