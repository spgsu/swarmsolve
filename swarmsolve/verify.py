"""Outcome-based verification: what actually changed on disk, and whether the
project's own build/test command actually passes -- grounding the judge in
real state instead of only the agents' claims about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from swarmsolve import repo
from swarmsolve.tools import ToolPolicy, Workspace, run_shell

# (marker file, build_cmd or None, test_cmd) -- first match wins, in this order
_MARKERS: list[tuple[str, str | None, str]] = [
    ("package.json", "npm run build --if-present", "npm test --if-present"),
    ("Cargo.toml", "cargo build", "cargo test"),
    ("go.mod", "go build ./...", "go test ./..."),
    ("pyproject.toml", None, "python -m pytest -q"),
    ("setup.py", None, "python -m pytest -q"),
    ("pytest.ini", None, "python -m pytest -q"),
]

_MAKE_TEST_RE = re.compile(r"(?m)^test\s*:")
_MAKE_BUILD_RE = re.compile(r"(?m)^build\s*:")


@dataclass
class VerificationReport:
    files_changed: list[str] = field(default_factory=list)
    files_changed_known: bool = False  # False when there's no git base to diff against
    build_cmd: str | None = None
    build_ok: bool | None = None  # None = not run (not detected, or shell disabled)
    build_output: str = ""
    test_cmd: str | None = None
    test_ok: bool | None = None
    test_output: str = ""

    @property
    def has_failure(self) -> bool:
        return self.build_ok is False or self.test_ok is False

    def summary(self) -> str:
        parts: list[str] = []
        if self.files_changed_known:
            head = ", ".join(self.files_changed[:20])
            extra = f" (+{len(self.files_changed) - 20} more)" if len(self.files_changed) > 20 else ""
            parts.append(f"{len(self.files_changed)} file(s) changed: {head}{extra}" if self.files_changed
                         else "0 files changed")
        else:
            parts.append("files changed: unknown (no git base to diff against)")

        def _status(ok: bool | None) -> str:
            return "OK" if ok else "FAILED" if ok is False else "not run"

        if self.build_cmd:
            parts.append(f"build [{self.build_cmd}]: {_status(self.build_ok)}")
        if self.test_cmd:
            parts.append(f"tests [{self.test_cmd}]: {_status(self.test_ok)}")
        return " | ".join(parts)


def detect_commands(workspace_root: Path) -> tuple[str | None, str | None]:
    """Best-effort (build_cmd, test_cmd) from common project marker files.
    Neither is a guess at correctness -- just "does this ecosystem's usual
    command exist"; ``VerificationReport.build_ok``/``test_ok`` stay ``None``
    (not a failure) whenever nothing was detected."""
    for marker, build_cmd, test_cmd in _MARKERS:
        if (workspace_root / marker).is_file():
            return build_cmd, test_cmd
    tests_dir = workspace_root / "tests"
    if tests_dir.is_dir() and any(tests_dir.rglob("test_*.py")):
        return None, "python -m pytest -q"
    makefile = workspace_root / "Makefile"
    if makefile.is_file():
        try:
            content = makefile.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        build_cmd = "make build" if _MAKE_BUILD_RE.search(content) else None
        test_cmd = "make test" if _MAKE_TEST_RE.search(content) else None
        if build_cmd or test_cmd:
            return build_cmd, test_cmd
    return None, None


def run_verification(
    workspace: Workspace,
    policy: ToolPolicy,
    base_sha: str | None = None,
    build_cmd: str | None = None,
    test_cmd: str | None = None,
) -> VerificationReport:
    report = VerificationReport()

    if base_sha and repo.is_git_repo(workspace.root):
        report.files_changed = repo.diff_files(workspace.root, base_sha)
        report.files_changed_known = True

    detected_build, detected_test = detect_commands(workspace.root)
    build_cmd = build_cmd or detected_build
    test_cmd = test_cmd or detected_test

    if build_cmd:
        report.build_cmd = build_cmd
        if policy.allow_shell:
            res = run_shell(workspace, build_cmd, policy)
            report.build_ok = res.ok
            report.build_output = res.short(1500)
        else:
            report.build_output = "shell disabled (--no-shell); cannot run the detected build command"

    if test_cmd:
        report.test_cmd = test_cmd
        if policy.allow_shell:
            res = run_shell(workspace, test_cmd, policy)
            report.test_ok = res.ok
            report.test_output = res.short(1500)
        else:
            report.test_output = "shell disabled (--no-shell); cannot run the detected test command"

    return report
