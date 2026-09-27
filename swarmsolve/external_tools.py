"""Add a tool to the swarm's toolbox by writing a TOML file -- no Python required.

:mod:`swarmsolve.toolbox` already lets any *code* register a tool with one
decorator. This module is the no-code path onto the same registry: drop a
``.toml`` manifest describing an existing CLI program or HTTP endpoint into a
directory, call :func:`load_manifests` on that directory, and it dispatches
through ``TOOLBOX`` exactly like a built-in -- same prompt rendering, same
event-log wiring, because it's the same registry.

A manifest can only *call* something that already exists (a program on PATH,
an HTTP API), it can't express new logic. That trade shows up as a safety
win: no code executes from inside a manifest, so there is no sandbox to
escape -- the worst a manifest can do is run the CLI program it names with
the arguments it names, or call the HTTP endpoint it names.

Two manifest shapes
--------------------

CLI -- an argv template, executed directly (never through a shell, so shell
metacharacters in a parameter value can't break out of their argument)::

    name = "line_count"
    description = "Count non-blank lines in a workspace file"
    kind = "cli"
    argv = ["wc", "-l", "{path}"]

    [params.path]
    required = true
    help = "relative/path"

HTTP -- a method + URL template, ``{param}`` filled into the path, other
params sent as query string (or JSON body for POST)::

    name = "http_status"
    description = "Fetch a fixed API's status endpoint"
    kind = "http"
    method = "GET"
    url = "https://api.example.com/status/{id}"

    [params.id]
    required = true
    in = "path"

    [params.verbose]
    in = "query"
"""

from __future__ import annotations

import re
import shutil
import string
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarmsolve.toolbox import TOOLBOX, Toolbox
from swarmsolve.tools import _SHELL_DENY, ToolError, ToolPolicy, ToolResult, Workspace, _guard_url

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python < 3.11
    tomllib = None  # type: ignore[assignment]

_FMT = string.Formatter()


@dataclass
class ParamSpec:
    name: str
    required: bool = False
    help: str = ""
    where: str = "query"  # http only: "query" | "path" | "body"


@dataclass
class ManifestTool:
    name: str
    description: str
    kind: str  # "cli" | "http"
    params: dict[str, ParamSpec] = field(default_factory=dict)
    # cli
    argv: list[str] = field(default_factory=list)
    # http
    method: str = "GET"
    url: str = ""


class ManifestError(ValueError):
    """A .toml file is not a valid tool manifest."""


def _parse_params(raw: dict[str, Any]) -> dict[str, ParamSpec]:
    out: dict[str, ParamSpec] = {}
    for pname, pspec in (raw or {}).items():
        pspec = pspec or {}
        if not isinstance(pspec, dict):
            raise ManifestError(f"params.{pname} must be a table")
        out[pname] = ParamSpec(
            name=pname,
            required=bool(pspec.get("required", False)),
            help=str(pspec.get("help", "")),
            where=str(pspec.get("in", "query")),
        )
    return out


def parse_manifest(path: Path) -> ManifestTool:
    if tomllib is None:  # pragma: no cover
        raise ManifestError("TOML manifests need Python 3.11+ (stdlib tomllib)")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ManifestError(f"invalid TOML: {e}") from e

    name = str(raw.get("name", "")).strip()
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ManifestError("'name' must be an identifier-like string")
    description = str(raw.get("description", "")).strip()
    if not description:
        raise ManifestError("'description' is required")
    kind = str(raw.get("kind", "")).strip()
    if kind not in ("cli", "http"):
        raise ManifestError("'kind' must be \"cli\" or \"http\"")
    params = _parse_params(raw.get("params", {}))

    if kind == "cli":
        argv = raw.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(t, str) for t in argv):
            raise ManifestError("'argv' must be a non-empty list of strings for kind=\"cli\"")
        return ManifestTool(name, description, kind, params, argv=list(argv))

    url = str(raw.get("url", "")).strip()
    if not url:
        raise ManifestError("'url' is required for kind=\"http\"")
    method = str(raw.get("method", "GET")).strip().upper()
    if method not in ("GET", "POST"):
        raise ManifestError("'method' must be GET or POST")
    return ManifestTool(name, description, kind, params, method=method, url=url)


def _args_help(tool: ManifestTool) -> str:
    if not tool.params:
        return "{}"
    parts = [f'"{p.name}": "{p.help or ("required" if p.required else "optional")}"' for p in tool.params.values()]
    return "{" + ", ".join(parts) + "}"


def _missing_required(tool: ManifestTool, args: dict) -> list[str]:
    return [p.name for p in tool.params.values() if p.required and not str(args.get(p.name, "")).strip()]


def _fill_argv(tool: ManifestTool, args: dict) -> list[str] | str:
    """Substitute {param} tokens; a whole-token placeholder takes the raw value verbatim
    (so a path with spaces stays one argv element). Returns an error string on failure.

    Optional params that the caller didn't provide are **dropped** — both the
    placeholder token and any preceding flag token (e.g. ``-e {engine}`` becomes
    nothing when ``engine`` is absent).  Required params that are missing still
    error.
    """
    required_names = {p.name for p in tool.params.values() if p.required}
    tokens = list(tool.argv)
    out: list[str] = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        fields = [f for _, f, _, _ in _FMT.parse(token) if f]
        if len(fields) == 1 and token == "{" + fields[0] + "}":
            # whole-token placeholder
            if fields[0] not in args or str(args[fields[0]]).strip() == "":
                if fields[0] in required_names:
                    return f"missing arg for {{{fields[0]}}}"
                # Optional param not provided — drop this token AND any
                # preceding flag token (e.g. the "-e" before "{engine}").
                if out and out[-1].startswith("-") and "{" not in out[-1]:
                    out.pop()
                continue
            out.append(str(args[fields[0]]))
            continue
        try:
            out.append(token.format(**{k: str(v) for k, v in args.items()}))
        except KeyError as e:
            field_name = str(e).strip("'\"")
            if field_name in required_names:
                return f"missing arg for {e}"
            # Optional inline placeholder — drop this token and any preceding flag
            if out and out[-1].startswith("-") and "{" not in out[-1]:
                out.pop()
    return out


def _make_cli_handler(tool: ManifestTool):
    def handler(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
        if not policy.allow_shell:
            return ToolResult(False, "shell/process execution is disabled (--no-shell)")
        missing = _missing_required(tool, args)
        if missing:
            return ToolResult(False, f"missing required arg(s): {', '.join(missing)}")
        filled = _fill_argv(tool, args)
        if isinstance(filled, str):
            return ToolResult(False, filled)
        joined = " ".join(filled)
        if _SHELL_DENY.search(joined):
            return ToolResult(False, f"refused: matches the destructive-action denylist\n  {joined}")
        if shutil.which(filled[0]) is None and not Path(filled[0]).exists():
            return ToolResult(False, f"'{filled[0]}' is not on PATH")
        try:
            proc = subprocess.run(
                filled, cwd=str(ws.root), capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=policy.shell_timeout, shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"command timed out after {policy.shell_timeout:g}s", {"timeout": True})
        except (FileNotFoundError, OSError) as e:
            return ToolResult(False, f"cannot run {filled[0]}: {e}")
        body = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        if len(body) > policy.max_output:
            body = body[: policy.max_output] + f"\n... [output capped at {policy.max_output} chars]"
        return ToolResult(proc.returncode == 0, f"$ {joined}\n(exit {proc.returncode})\n{body}",
                          {"returncode": proc.returncode})

    return handler


def _make_http_handler(tool: ManifestTool):
    def handler(ws: Workspace, args: dict, policy: ToolPolicy) -> ToolResult:
        if not policy.allow_web:
            return ToolResult(False, "web access is disabled (--no-web)")
        missing = _missing_required(tool, args)
        if missing:
            return ToolResult(False, f"missing required arg(s): {', '.join(missing)}")

        path_params = {p.name: args[p.name] for p in tool.params.values() if p.where == "path" and p.name in args}
        query_params = {p.name: args[p.name] for p in tool.params.values() if p.where == "query" and p.name in args}
        body_params = {p.name: args[p.name] for p in tool.params.values() if p.where == "body" and p.name in args}

        try:
            url = tool.url.format(**{k: urllib.parse.quote(str(v), safe="") for k, v in path_params.items()})
        except KeyError as e:
            return ToolResult(False, f"missing path arg for {e}")
        if query_params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urllib.parse.urlencode({k: str(v) for k, v in query_params.items()})

        try:
            _guard_url(url)
        except ToolError as e:
            return ToolResult(False, str(e))

        import json as _json
        import urllib.error
        import urllib.request

        headers = {"User-Agent": "swarmsolve-toolbox/1.0"}
        data = None
        if tool.method == "POST":
            data = _json.dumps(body_params).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=tool.method)
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                raw = resp.read(policy.fetch_bytes)
                status = resp.status
        except urllib.error.HTTPError as e:
            return ToolResult(False, f"HTTP {e.code} calling {url}", {"status": e.code})
        except (urllib.error.URLError, TimeoutError) as e:
            return ToolResult(False, f"cannot call {url}: {e}")
        text = raw.decode("utf-8", errors="replace")[: policy.fetch_chars]
        return ToolResult(True, f"HTTP {status}\n\n{text}", {"status": status})

    return handler


def register_manifest(tool: ManifestTool, source: str, toolbox: Toolbox = TOOLBOX) -> None:
    handler = _make_cli_handler(tool) if tool.kind == "cli" else _make_http_handler(tool)
    from swarmsolve.toolbox import ToolSpec

    toolbox.register(ToolSpec(
        tool.name, _args_help(tool), handler,
        label=lambda a, n=tool.name: n,
        source=source,
        description=tool.description,
    ))


def load_manifests(directory: str | Path, toolbox: Toolbox = TOOLBOX) -> tuple[list[str], list[str]]:
    """Load every ``*.toml`` in ``directory`` into ``toolbox``. Idempotent -- a manifest
    already registered from the same file is left alone; a manifest whose name clashes
    with a built-in or a manifest from a different file is reported as an error. Never
    raises: a bad file is reported, not fatal, since a single typo shouldn't take down
    the swarm. Returns (names loaded this call, error messages)."""
    directory = Path(directory)
    loaded: list[str] = []
    errors: list[str] = []
    if not directory.is_dir():
        return loaded, errors
    for path in sorted(directory.glob("*.toml")):
        try:
            tool = parse_manifest(path)
            source = f"manifest:{path.resolve()}"
            existing = toolbox.get(tool.name)
            if existing is not None:
                if existing.source == source:
                    continue  # already loaded from this exact file — silent no-op
                errors.append(f"{path.name}: name {tool.name!r} already registered ({existing.source})")
                continue
            register_manifest(tool, source, toolbox)
            loaded.append(tool.name)
        except ManifestError as e:
            errors.append(f"{path.name}: {e}")
    return loaded, errors


def list_manifest_tools(directory: str | Path, toolbox: Toolbox = TOOLBOX) -> list[dict]:
    """Every currently-registered tool whose source is a manifest under ``directory``,
    plus a small metadata record per tool."""
    directory = Path(directory).resolve() if directory else None
    out: list[dict] = []
    for spec in toolbox.by_source("manifest:"):
        source_path = Path(spec.source.split("manifest:", 1)[1])
        # Every registered manifest is included, even if its file has been renamed
        # or moved elsewhere -- the source path is the live truth.
        under_dir = directory is not None and str(source_path).startswith(str(directory))
        out.append({
            "name": spec.name,
            "description": spec.description,
            "args_help": spec.args_help,
            "source_file": source_path.name if source_path else "",
            "source_path": str(source_path) if source_path else "",
            "in_dir": under_dir,
        })
    return out


def save_manifest(directory: str | Path, filename: str, content: str,
                  toolbox: Toolbox = TOOLBOX) -> dict:
    """Validate a TOML manifest, write it into ``directory``, and register it.
    Returns ``{"name": ..., "path": ...}`` on success; raises ``ManifestError``
    for a broken manifest or a name/file conflict, without touching the disk or
    the registry."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(filename)
    target = directory / filename
    if target.exists():
        raise ManifestError(f"{filename} already exists in the toolbox directory")

    # Validate FIRST -- parse from a temp path so a broken manifest never touches
    # the toolbox directory. Only write once we know it's good.
    tmp = directory / (filename + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    try:
        tool = parse_manifest(tmp)
    except ManifestError:
        tmp.unlink(missing_ok=True)
        raise
    source = f"manifest:{target.resolve()}"
    existing = toolbox.get(tool.name)
    if existing is not None:
        tmp.unlink(missing_ok=True)
        raise ManifestError(f"tool name {tool.name!r} is already registered ({existing.source})")
    tmp.rename(target)
    register_manifest(tool, source, toolbox)
    return {"name": tool.name, "path": str(target), "description": tool.description}


def remove_manifest(directory: str | Path, name: str,
                    toolbox: Toolbox = TOOLBOX) -> dict:
    """Unregister a manifest tool and delete its file (only if the file lives in
    ``directory`` -- a manifest loaded from elsewhere is only unregistered)."""
    directory = Path(directory).resolve()
    spec = toolbox.get(name)
    if spec is None or not spec.source.startswith("manifest:"):
        raise ManifestError(f"tool {name!r} is not a manifest tool")
    source_path = Path(spec.source.split("manifest:", 1)[1])
    deleted = False
    try:
        source_path.resolve().relative_to(directory)
        source_path.unlink(missing_ok=True)
        deleted = True
    except ValueError:
        pass  # file lives outside our directory — only unregister
    toolbox.remove(name)
    return {"name": name, "file_deleted": deleted, "path": str(source_path)}


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

def _safe_filename(name: str) -> str:
    """Reject anything that could escape the toolbox directory (path separators,
    empty strings) or shadow an OS-special filename. Never raises the user's
    typo into a security incident."""
    name = (name or "").strip()
    if not name:
        raise ManifestError("filename is required")
    if not name.lower().endswith(".toml"):
        name += ".toml"
    # Strip any directory components the user may have snuck in.
    from os.path import basename
    name = basename(name)
    if _SAFE_NAME.search(name):
        raise ManifestError(f"filename may only contain letters, digits, '.', '_', '-'; got {name!r}")
    if name.startswith(".") or name.startswith("-"):
        raise ManifestError(f"filename must not start with '.' or '-'; got {name!r}")
    return name
