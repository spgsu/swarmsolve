"""Sandboxed capabilities for the task swarm: shell, files, web.

Everything an agent can *do* to the outside world goes through here. Two guard
rails:

* **Filesystem** -- every path is resolved and checked to live under the per-run
  workspace directory (:func:`safe_path`). No absolute paths, no ``..`` escape,
  no following a symlink out.
* **Shell** -- run with ``cwd`` = workspace, a wall-clock timeout, output capped.
  Wrapped in ``bwrap`` (bubblewrap, Linux-only) when available for real fs/network
  isolation; otherwise a regex denylist (POSIX and Windows spellings both) refuses
  the obviously destructive stuff. PowerShell on native Windows, ``/bin/sh``
  elsewhere (:func:`_shell_argv`). This is mitigation, not a jail -- an agent with
  shell can still do damage inside the workspace by design, and there is currently
  no bwrap-equivalent sandbox on Windows/Mac at all, denylist only. Native
  Windows uses PowerShell rather than ``cmd.exe`` so quoted subprocess
  commands and the PowerShell syntax advertised to workers survive intact.
* **Web** -- ``fetch_url`` refuses non-http(s), localhost, link-local and private
  addresses (SSRF guard). ``web_search`` scrapes DuckDuckGo's HTML endpoint
  (free, no key, brittle to markup changes).

Tools never raise into the pipeline: they return a :class:`ToolResult`.
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# refused outright -- catastrophic or host-level actions with no place in a workspace task.
# Covers both POSIX and Windows spellings: run_shell() executes on Windows too
# (PowerShell, see _shell_argv) rather than failing outright on a missing /bin/sh, so
# this denylist being POSIX-only would have been a real safety regression on that platform.
_SHELL_DENY = re.compile(
    r"""
      \brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|/\*)      # rm -rf / , rm -rf ~
    | (^|[\s;&|]):\s*\(\s*\)\s*\{                      # fork bomb  :(){ :|:& };:
    | \w+\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;       # named fork bomb
    | \bmkfs(\.\w+)?\b
    | \bdd\b[^\n]*\bof=/dev/                           # dd of=/dev/sda
    | \bsudo\b | \bsu\s | \bdoas\b
    | \b(shutdown|reboot|halt|poweroff|init\s+0)\b
    | \b(chown|chmod)\s+-R?\s*[^\n]*\s+/(?!\w*workspace)   # recursive perms on real roots
    | >\s*/dev/sd | >\s*/dev/nvme
    | \b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba)?sh     # curl ... | sh
    | \bnc\b[^\n]*-e                                   # netcat reverse shell
    | \bformat\s+[a-z]:                                # format C:
    | \bdel\s+(/[a-z]\s+)*[a-z]:\\                      # del /f /s /q C:\...
    | \b(rd|rmdir)\s+(/[a-z]\s+)*[a-z]:\\               # rd /s /q C:\...
    | %0\s*\|\s*%0                                      # windows batch fork bomb %0|%0
    | \breg\s+delete\b
    | \bdiskpart\b
    | \bvssadmin\s+delete\b                             # shadow-copy/backup wipe
    | \bcipher\s+/w:                                    # secure-wipe free space
    """,
    re.VERBOSE | re.IGNORECASE,
)


class ToolError(Exception):
    """A tool was asked to do something disallowed (path escape, blocked URL...)."""


def _shell_argv(cmd: str) -> list[str]:
    """The one-shot "run this command line" invocation for the current OS.
    bwrap (Linux-only, gated separately by shutil.which) always wraps a POSIX
    shell; without it, native Windows Python has no /bin/sh at all. PowerShell
    is the native worker contract on Windows; passing the command as its own
    argv item also avoids cmd.exe stripping the inner quotes from commands such
    as ``python -c \"print(1)\"``."""
    if sys.platform == "win32":
        return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd]
    return ["/bin/sh", "-c", cmd]


@dataclass
class ToolPolicy:
    allow_shell: bool = True
    allow_web: bool = True
    allow_write: bool = True
    bwrap: bool = True
    shell_timeout: float = 60.0
    max_output: int = 12_000
    fetch_bytes: int = 512 * 1024
    fetch_chars: int = 8_000

    def describe(self) -> str:
        bw = "bwrap" if (self.bwrap and shutil.which("bwrap")) else "no-bwrap"
        return (
            f"shell={'on' if self.allow_shell else 'off'} ({bw}), "
            f"web={'on' if self.allow_web else 'off'}, "
            f"write={'on' if self.allow_write else 'off'}"
        )


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict | None = None

    def short(self, n: int = 500) -> str:
        if len(self.output) <= n:
            return self.output
        if not self.ok and n >= 120:
            # A traceback's exception and message are at the end. Keeping only
            # the prefix made Windows failures especially opaque because
            # PowerShell's NativeCommandError wrapper consumed the whole GUI
            # preview before the actual Python exception appeared.
            marker = f"\n... [{len(self.output) - n} chars omitted; failure tail] ...\n"
            room = max(40, n - len(marker))
            head = room // 3
            return self.output[:head] + marker + self.output[-(room - head):]
        return self.output[:n] + f"... [+{len(self.output) - n} chars]"


class Workspace:
    """A directory the swarm may read and write. Created fresh per run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def safe_path(self, rel: str) -> Path:
        return safe_path(self.root, rel)

    def list_files(self) -> list[str]:
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                # Stable workspace-relative paths are part of the worker protocol.
                # Always use '/' so Windows runs do not leak backslash variants
                # into prompts or blackboard comparisons.
                out.append(p.relative_to(self.root).as_posix())
        return out

    def summary(self, max_files: int = 40) -> str:
        files = self.list_files()
        head = files[:max_files]
        extra = "" if len(files) <= max_files else f" (+{len(files) - max_files} more)"
        return ", ".join(head) + extra if head else "(empty)"


def safe_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; raise ToolError if it would escape."""
    root = root.resolve()
    if not rel or not str(rel).strip():
        raise ToolError("empty path -- pass a path relative to the workspace root, e.g. 'src/app.py'")
    is_abs = Path(rel).is_absolute()
    candidate = (root / rel).resolve() if not is_abs else Path(rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        # Targeted hint on the common failure mode: an LLM copies an absolute
        # path out of the task text ("C:\\Users\\...\\the-project\\app.py") into
        # a read_file/write_file arg. The project HAS been checked out into the
        # workspace by isolation -- they should use the tail of the path relative
        # to the workspace root, not the original absolute one.
        if is_abs:
            tail = Path(rel).name
            raise ToolError(
                f"path {rel!r} is absolute and points outside the workspace ({root}). "
                f"Use a path RELATIVE to the workspace root instead -- for that file, "
                f"probably just {tail!r} (or 'list_files' / 'shell git ls-tree -r HEAD' "
                f"to see what's actually here)."
            ) from None
        raise ToolError(f"path {rel!r} escapes the workspace ({root})") from None
    return candidate


# -- files ------------------------------------------------------------------

def _read_project_text(path: Path) -> tuple[str, str]:
    """Decode common project encodings and return a safe rewrite encoding."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"

def read_file(ws: Workspace, path: str, max_chars: int = 20_000) -> ToolResult:
    try:
        p = ws.safe_path(path)
    except ToolError as e:
        return ToolResult(False, str(e))
    if not p.is_file():
        return ToolResult(False, f"no such file: {path}")
    try:
        text, encoding = _read_project_text(p)
    except OSError as e:
        return ToolResult(False, f"cannot read {path}: {e}")
    truncated = len(text) > max_chars
    return ToolResult(
        True,
        text[:max_chars] + ("\n... [truncated]" if truncated else ""),
        {"bytes": p.stat().st_size, "truncated": truncated, "encoding": encoding},
    )


def write_file(ws: Workspace, path: str, content: str, policy: ToolPolicy) -> ToolResult:
    if not policy.allow_write:
        return ToolResult(False, "file writes are disabled (--no-write)")
    try:
        p = ws.safe_path(path)
    except ToolError as e:
        return ToolResult(False, str(e))
    new_content = content if content is not None else ""
    encoding = "utf-8"
    if p.is_file():
        try:
            existing, encoding = _read_project_text(p)
            if existing == new_content:
                return ToolResult(True, f"unchanged: {path}", {"path": str(p), "changed": False})
        except OSError:
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_content, encoding=encoding)
    return ToolResult(True, f"wrote {len(new_content)} chars to {path}", {"path": str(p), "changed": True})


def replace_text(
    ws: Workspace, path: str, old: str, new: str, policy: ToolPolicy, expected_count: int = 1,
) -> ToolResult:
    """Atomically replace an exact, uniquely-identified text block.

    This is the safe small-edit counterpart to ``generate_file``. Refusing a
    count mismatch prevents a stale model instruction from silently editing the
    wrong occurrence, and Python is syntax-checked before the original moves.
    """
    if not policy.allow_write:
        return ToolResult(False, "file writes are disabled (--no-write)")
    if not old:
        return ToolResult(False, "replace_text requires a non-empty old block")
    try:
        expected = int(expected_count)
    except (TypeError, ValueError):
        return ToolResult(False, "expected_count must be an integer")
    if expected < 1:
        return ToolResult(False, "expected_count must be at least 1")
    try:
        p = ws.safe_path(path)
    except ToolError as e:
        return ToolResult(False, str(e))
    if not p.is_file():
        return ToolResult(False, f"no such file: {path}")
    try:
        original, encoding = _read_project_text(p)
    except OSError as e:
        return ToolResult(False, f"cannot read {path}: {e}")
    found = original.count(old)
    if found != expected:
        return ToolResult(
            False,
            f"replace_text expected {expected} occurrence(s) in {path}, found {found}; file left unchanged",
        )
    updated = original.replace(old, new, expected)
    if updated == original:
        return ToolResult(True, f"unchanged: {path}", {"path": str(p), "count": found, "changed": False})
    if p.suffix.lower() == ".py":
        try:
            compile(updated, path, "exec")
        except SyntaxError as e:
            return ToolResult(
                False,
                f"replacement would make Python invalid at line {e.lineno}: {e.msg}; file left unchanged",
            )
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding=encoding, newline="", dir=p.parent, prefix=f".{p.name}.", delete=False,
        ) as fh:
            fh.write(updated)
            tmp_name = fh.name
        os.replace(tmp_name, p)
    except OSError as e:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        return ToolResult(False, f"cannot update {path}: {e}")
    return ToolResult(
        True, f"replaced {found} exact block(s) in {path}",
        {"path": str(p), "count": found, "changed": True},
    )


def list_files(ws: Workspace) -> ToolResult:
    files = ws.list_files()
    return ToolResult(True, "\n".join(files) if files else "(workspace is empty)", {"count": len(files)})


# -- shell ----------------------------------------------------------------

def run_shell(ws: Workspace, cmd: str, policy: ToolPolicy) -> ToolResult:
    if not policy.allow_shell:
        return ToolResult(False, "shell is disabled (--no-shell)")
    if not cmd or not cmd.strip():
        return ToolResult(False, "empty command")
    if _SHELL_DENY.search(cmd):
        return ToolResult(False, f"refused: command matches the destructive-action denylist\n  {cmd}")

    argv: list[str]
    use_bwrap = policy.bwrap and shutil.which("bwrap") is not None
    if use_bwrap:
        argv = [
            "bwrap", "--ro-bind", "/", "/",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", str(ws.root), str(ws.root),
            "--chdir", str(ws.root),
            "--die-with-parent", "--new-session",
        ]
        if not policy.allow_web:
            argv += ["--unshare-net"]
        argv += _shell_argv(cmd)
    else:
        argv = _shell_argv(cmd)

    try:
        proc = subprocess.run(
            argv,
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=policy.shell_timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"command timed out after {policy.shell_timeout:g}s", {"timeout": True})
    except FileNotFoundError as e:
        return ToolResult(False, f"cannot run shell: {e}")

    body = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if len(body) > policy.max_output:
        body = body[: policy.max_output] + f"\n... [output capped at {policy.max_output} chars]"
    return ToolResult(
        proc.returncode == 0,
        f"$ {cmd}\n(exit {proc.returncode}{', bwrap' if use_bwrap else ''})\n{body}",
        {"returncode": proc.returncode, "bwrap": use_bwrap},
    )


# -- web ----------------------------------------------------------------

def _host_is_private(host: str) -> bool:
    if host.lower() in ("localhost", "localhost.localdomain", ""):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False  # unresolvable -> let urllib fail naturally
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return True
    return False


def _guard_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"refused: only http(s) URLs allowed (got {parsed.scheme or 'no scheme'})")
    host = parsed.hostname or ""
    if _host_is_private(host):
        raise ToolError(f"refused: {host} resolves to a local/private address (SSRF guard)")
    return url


def fetch_url(url: str, policy: ToolPolicy) -> ToolResult:
    if not policy.allow_web:
        return ToolResult(False, "web access is disabled (--no-web)")
    try:
        _guard_url(url)
    except ToolError as e:
        return ToolResult(False, str(e))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            raw = resp.read(policy.fetch_bytes)
            ctype = resp.headers.get("Content-Type", "")
            status = resp.status
            headers = {k: v for k, v in resp.headers.items() if k.lower() in
                      ("server", "x-powered-by", "content-type", "www-authenticate", "location")}
    except urllib.error.HTTPError as e:
        return ToolResult(False, f"HTTP {e.code} fetching {url}", {"status": e.code})
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        return ToolResult(False, f"cannot fetch {url}: {e}")
    text = raw.decode("utf-8", errors="replace")
    if "html" in ctype.lower() or "<html" in text[:2000].lower():
        text = _strip_html(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = len(text) > policy.fetch_chars
    body = f"HTTP {status}  headers={headers}\n\n" + text[: policy.fetch_chars]
    return ToolResult(True, body + ("\n... [truncated]" if truncated else ""),
                      {"status": status, "headers": headers, "content_type": ctype, "truncated": truncated})


def web_search(query: str, policy: ToolPolicy, n: int = 5) -> ToolResult:
    if not policy.allow_web:
        return ToolResult(False, "web access is disabled (--no-web)")
    if not query or not query.strip():
        return ToolResult(False, "empty query")

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            html = resp.read(400_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return ToolResult(False, f"search HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        return ToolResult(False, f"search failed: {e}")

    results = _parse_ddg(html, n)
    if not results:
        return ToolResult(False, "no results parsed (DuckDuckGo markup may have changed)")
    lines = [f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results)]
    return ToolResult(True, "\n".join(lines), {"results": results})


_A_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_SNIP_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _detag(s: str) -> str:
    import html as _html

    return _html.unescape(_TAG_RE.sub("", s)).strip()


def _parse_ddg(html: str, n: int) -> list[dict]:
    hrefs = _A_RE.findall(html)
    snips = _SNIP_RE.findall(html)
    out: list[dict] = []
    for i, (href, title) in enumerate(hrefs[:n]):
        real = href
        m = re.search(r"[?&]uddg=([^&]+)", href)
        if m:
            real = urllib.parse.unquote(m.group(1))
        out.append({
            "title": _detag(title),
            "url": real,
            "snippet": _detag(snips[i]) if i < len(snips) else "",
        })
    return out


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|head|nav|footer|svg)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr)>", "\n", html)
    return _detag(html)
