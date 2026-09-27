"""Dependency-free ``.env`` loader.

Reads ``KEY=VALUE`` lines (``#`` comments, blank lines, and ``export`` prefixes
tolerated) and sets them in ``os.environ`` *without* overwriting variables already
present. Call :func:`load_env` before constructing any API-backed substrate; the
runner and the LLM substrates already do.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded: set[str] = set()


def load_env(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    p = Path(path)
    key = str(p.resolve())
    if key in _loaded and not override:
        return {}
    applied: dict[str, str] = {}
    if not p.is_file():
        _loaded.add(key)
        return applied
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if not name:
            continue
        if override or name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    _loaded.add(key)
    return applied
