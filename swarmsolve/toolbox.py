"""A registry any capability can be added to without touching the dispatch loop.

Before this, adding a static tool meant three hand-edits in lockstep: a function
in :mod:`swarmsolve.tools`, an ``elif action == "..."`` branch in
:class:`swarmsolve.task_env.OpenTaskEnv`, and a matching line in the worker
prompt in :mod:`swarmsolve.prompts`. Miss one and the tool either can't be
reached or the model doesn't know it exists.

Now: write the handler, decorate it with :data:`TOOLBOX`, done. ``task_env.py``
dispatches any registered name generically; ``prompts.py`` renders the args-help
block from the same registry, so the two can't drift apart.

Scope: this covers stateless-ish capabilities shaped like the existing ones --
``(Workspace, args, ToolPolicy) -> ToolResult``, never raising.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from swarmsolve.tools import ToolPolicy, ToolResult, Workspace

Handler = Callable[[Workspace, dict, ToolPolicy], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_help: str  # shown verbatim in the worker prompt, e.g. '{"path": "relative/path"}'
    handler: Handler
    enabled: Callable[[ToolPolicy], bool] = lambda policy: True  # noqa: E731 -- hide from the prompt when false
    label: Callable[[dict], str] = lambda args: ""  # noqa: E731 -- event-log tag, e.g. "read {path}"
    source: str = "code"  # provenance: "code" for @tool decorator, "manifest:<path>" for a TOML manifest
    description: str = ""  # human-facing summary, shown in listings (empty for @tool-registered built-ins)


class Toolbox:
    """Name -> :class:`ToolSpec`. One instance (:data:`TOOLBOX`) is the live registry."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        args_help: str,
        *,
        enabled: Callable[[ToolPolicy], bool] = lambda policy: True,  # noqa: E731
        label: Callable[[dict], str] = lambda args: "",  # noqa: E731
    ) -> Callable[[Handler], Handler]:
        """Decorator form: ``@TOOLBOX.tool("name", '{"arg": "..."}')`` above a handler."""

        def decorate(fn: Handler) -> Handler:
            self.register(ToolSpec(name, args_help, fn, enabled, label))
            return fn

        return decorate

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def remove(self, name: str) -> ToolSpec | None:
        """Remove a registered tool by name, if present. Returns the removed spec or None."""
        return self._specs.pop(name, None)

    def by_source(self, prefix: str) -> list[ToolSpec]:
        """All specs whose ``source`` starts with ``prefix`` — used by the manifest
        loader to find which registered tools it owns for a clean reload."""
        return [s for s in self._specs.values() if s.source.startswith(prefix)]

    def names(self, policy: ToolPolicy | None = None, *, exclude: frozenset[str] = frozenset()) -> list[str]:
        """Registered names, in registration order, filtered to those enabled for ``policy``.

        ``exclude`` is for callers that only know a proxy for policy (e.g. the worker
        prompt only has a scope string, not a full ``ToolPolicy``) -- name a tool whose
        ``enabled`` predicate can't be evaluated yet and it's left out of this listing.
        """
        return [n for n, s in self._specs.items() if n not in exclude and (policy is None or s.enabled(policy))]

    def custom_tools(self, *, exclude: frozenset[str] = frozenset()) -> list[ToolSpec]:
        """Manifest-registered tools (source starts with 'manifest:'), separate from built-ins."""
        return [s for s in self._specs.values() if s.source.startswith("manifest:") and s.name not in exclude]

    def args_help(self, name: str) -> str:
        return self._specs[name].args_help

    def prompt_lines(self, policy: ToolPolicy | None = None, *, exclude: frozenset[str] = frozenset()) -> list[str]:
        return [f"  {name:<13}{self._specs[name].args_help}" for name in self.names(policy, exclude=exclude)]

    def label_for(self, name: str, args: dict) -> str:
        spec = self._specs.get(name)
        tag = spec.label(args) if spec else ""
        return tag or name

    def dispatch(self, name: str, ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(False, f"unknown tool {name!r}")
        if not spec.enabled(policy):
            return ToolResult(False, f"tool {name!r} is disabled for this run")
        return spec.handler(ws, args, policy)


TOOLBOX = Toolbox()


# -- built-ins: thin wrappers registering the existing swarmsolve.tools functions -----

from swarmsolve import tools as _tools  # noqa: E402 -- after Toolbox so decorators can use it


@TOOLBOX.tool("list_files", "{}", label=lambda a: "files")
def _list_files(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.list_files(ws)


@TOOLBOX.tool(
    "read_file", '{"path": "relative/path"}',
    label=lambda a: f"read {a.get('path')}",
)
def _read_file(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.read_file(ws, str(args.get("path", "")))


@TOOLBOX.tool(
    "write_file", '{"path": "relative/path", "content": "full file contents"}',
    label=lambda a: "write",
)
def _write_file(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.write_file(ws, str(args.get("path", "")), str(args.get("content", "")), policy)


@TOOLBOX.tool(
    "replace_text",
    '{"path": "relative/path", "old": "exact existing block", "new": "replacement", "expected_count": 1}',
    label=lambda a: f"replace {a.get('path')}",
)
def _replace_text(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.replace_text(
        ws,
        str(args.get("path", "")),
        str(args.get("old", "")),
        str(args.get("new", "")),
        policy,
        args.get("expected_count", 1),
    )


@TOOLBOX.tool(
    "shell", '{"cmd": "shell command, runs in the workspace dir"}',
    label=lambda a: "shell",
)
def _shell(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    cmd = str(args.get("cmd", ""))
    # Worker shell is intentionally inspection/test oriented. Source rewriting
    # through nested shell/Python quoting caused repeated syntax corruption on
    # Windows. Verification still calls run_shell() directly, so build tools can
    # create their normal artifacts; only agent-authored edit commands are routed
    # to the atomic generate_file/write_file paths.
    edit_patterns = (
        r"\bset-content\b", r"\badd-content\b", r"\bout-file\b",
        r"\bsed\s+-i\b", r"\bperl\s+-pi\b",
        r"\.write_text\s*\(", r"\.write_bytes\s*\(", r"\.writelines?\s*\(",
        r"\[io\.file\]::write", r"\btee\b",
    )
    if any(re.search(pattern, cmd, re.IGNORECASE) for pattern in edit_patterns):
        return ToolResult(
            False,
            "source editing through shell is disabled because quoting/indentation is unreliable; "
            "use generate_file for a substantial rewrite, replace_text for a targeted edit, "
            "or write_file for a small complete file",
        )
    return _tools.run_shell(ws, cmd, policy)


@TOOLBOX.tool(
    "web_search", '{"query": "search terms"}',
    label=lambda a: f"search {a.get('query')}",
)
def _web_search(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.web_search(str(args.get("query", "")), policy)


@TOOLBOX.tool(
    "fetch_url", '{"url": "https://..."}',
    label=lambda a: f"fetch {a.get('url')}",
)
def _fetch_url(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
    return _tools.fetch_url(str(args.get("url", "")), policy)
